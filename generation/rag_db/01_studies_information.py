import pandas as pd 
import os
from openai import OpenAI
from PyPDF2 import PdfReader
import collections
from models import models
from utils_call_models import query_model_structured_message
from utils import *

# Pdf files 
pdf_path_1 = "studies/PrimaryTumor/"
pdf_path_2 = "studies/LiverMetastases/"
pdf_path_3 = "studies/ExtraHepaticMetastases/"
pdf_path_4 = "studies/RectalPrimaryTumor/"

seen_files, pdf_files = [],[]
for path_ in [pdf_path_1, pdf_path_2, pdf_path_3, pdf_path_4]:
    file_list = os.listdir(path_)
    for f in file_list:
        if f[-3:]=="pdf" and f not in seen_files:
            pdf_files += [f"{path_}{f}"] 
            seen_files.append(f)
print(f"N. studies pdf: {len(pdf_files)}")
duplicates = [item for item, count in collections.Counter(pdf_files).items() if count > 1]
print(f"N. duplicated pdf: {len(duplicates)}")

# Extract the study information using LLM (gpt-oss-120B)
prompt_template = open("information_extraction_prompt.txt", "r").read()
client = OpenAI(api_key="empty", base_url=models["llm"]["url"])

studies_information = pd.DataFrame(columns=["File_Name", "First_Author_Surname", "Year_Publication", "Title", "Topic"])

for f in pdf_files:
    print(f"File name: {f}")
    file_output_reader = PdfReader(f)
    first_page = file_output_reader.pages[0].extract_text()
    prompt = prompt_template.format(study=first_page)
    study_information = query_model_structured_message(client, "", prompt)
    study_information_dict = extract_structured_output(study_information)
    if study_information_dict is not None:
        items = extract_items(study_information_dict, ["First_Author_Surname", "Year_Publication", "Title"])
        if len(items) == 3:
            if "RectalPrimaryTumor" in f:
                topic = "RectalPrimaryTumor"
            elif "PrimaryTumor" in f:
                topic = "PrimaryTumor"
            elif "LiverMetastases" in f:
                topic = "LiverMetastases"
            elif "ExtraHepaticMetastases" in f:
                topic = "ExtraHepaticMetastases"
            path_begin = len(f"studies/{topic}/")
            file_name = f[path_begin:]
            studies_information.loc[len(studies_information)] = [file_name, items[0], items[1], items[2], topic]
            print(f"Table len {len(studies_information)}\n==========================================================")

print(f"Processed files: {len(studies_information)}")
print(studies_information.iloc[-3:,:])
studies_information.to_csv("../data/studies_information.csv", index=False)