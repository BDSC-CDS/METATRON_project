import pymupdf4llm
import regex
import re
import pandas as pd 
import os
import collections
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Document
from utils import *

# FUNCTIONS
# Split the mardown using symbol "*"
def split_markdown(md_text: str, paragraphs_tb: pd.DataFrame, ref_id: int, title: str, author: str, topic: str, par_len_t: int):
    par_list = md_text.split("**")
    par_list = [item for item, count in collections.Counter(par_list).items() if count == 1]
    par_counter = 1
    for par in list(par_list):
        if len(par) > par_len_t and par.replace("\n\n", "\n") not in paragraphs_tb["Content"].to_list():
            paragraphs_tb.loc[len(paragraphs_tb)] = [ref_id + par_counter, title, author, par_counter, par.replace("\n\n", "\n"), topic]
            par_counter += 1
    return paragraphs_tb

# Chunks splitter (to be used when paragrpahs are not found)
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
def split_in_chunks(md_text: str, paragraphs_tb: pd.DataFrame, ref_id: int, title: str, author: str, topic: str):
    doc = Document(text=md_text)
    nodes = splitter.get_nodes_from_documents([doc])
    if len(nodes) > 0:
        par_counter = 1
        for n in nodes:
            content = n.text.replace("\n\n", "\n")
            paragraphs_tb.loc[len(paragraphs_tb)] = [ref_id + par_counter, title, author, par_counter, content, topic]
            par_counter += 1
    else:
        print("Nodes were not found!")
    return paragraphs_tb

# Try splitting strategies 
def check_split_strategy(md_text: str, ref_id: int, title: str, author: str, topic: str, tb_columns: list, par_len_t: int, n_par_t: int):
    paragraphs_tb = split_markdown(md_text, pd.DataFrame(columns=tb_columns), ref_id, title, author, topic, par_len_t)
    if len(paragraphs_tb) < n_par_t:
        print("Not enough paragraphs: use chunks")
        paragraphs_tb = split_in_chunks(md_text, pd.DataFrame(columns=paragraphs_tb.columns), ref_id, title, author, topic)
    else:
        print("Successfull split in paragraphs using md symbols!")
    return paragraphs_tb

#####################################################################################################################

# Pdf files 
# Remove duplicates studies 
studies_information = pd.read_csv("../data/studies_information.csv")
print(f"Initial no. studies: {len(studies_information)}")
studies_information_unique = studies_information.drop_duplicates(subset=["Title"], keep="first").reset_index()
print(f"No. studies after removing duplicated titles: {len(studies_information_unique)}")
# Save the final table
studies_information_unique.to_csv("studies/studies_information_final.csv", index=False)
# PDF files 
pdf_files = []
for i in range(0, len(studies_information_unique)):
    pdf_name = studies_information_unique.loc[i,"File_Name"]
    pdf_files.append(pdf_name)
print(f"No. pdf files: {len(pdf_files)}")

# Split the pdf content into paragraphs 
# Paragraph pattern: N. [....]
paragraph_pattern = [r"\d+?\.\s+(.*?)[ \t]*(?=\r?\n(?:\r?\n)+)", r"\*\*.*?\*\*\s*"]
# Consider the study content from abstract to references 
start_pattern = "abstract"
end_pattern = "references"
# Set a threshold on the number of paragraphs indicating that a sufficient division was found
N_PAR_T = 5
# Set a threshold on the paragraph len to ensure sufficient content
PAR_LEN_T = 100

paragraphs_tb = pd.DataFrame(columns=["Reference_ID", "Title", "Author", "Paragraph_N", "Content", "Topic"])

# Markdown 
excluded_studies = 0
study_counter = 0
markdown_dir = "studies/md_files/"

for f in pdf_files:
    print(f"{study_counter}) File: {f} in process....")
    study_counter +=1 
    if len(paragraphs_tb) == 0:
        ref_id = 0 
    else:
        ref_id = paragraphs_tb.iloc[len(paragraphs_tb)-1, 0]
    
    # Study information
    try: 
        info_row = studies_information_unique.loc[studies_information_unique["File_Name"]==f,:]
        topic = info_row.Topic.item()
        title = info_row.Title.item()
        author = info_row.First_Author_Surname.item()
    except:
        print("Probably some problems occured in comparing the study title...\n===============================================================")
        excluded_studies += 1
        continue

    # Convert into markdown   
    md_text = pymupdf4llm.to_markdown(f"studies/{topic}/{f}")

    # Content start 
    start_match = regex.search(start_pattern, md_text, re.IGNORECASE)
    if start_match:
        md_text = md_text[start_match.start():]

    # Content end 
    end_match = regex.search(end_pattern, md_text, re.IGNORECASE)
    if end_match:
        md_text = md_text[0:end_match.start()]

    # Save md text 
    mdf = open(markdown_dir + f"{f[:-3]}md", "w")
    mdf.write(md_text)
    mdf.close()

    # Paragraphs content 
    paragraphs_title = re.finditer(paragraph_pattern[0], md_text, re.IGNORECASE)
    paragraphs_title_pos = []
    if paragraphs_title:
        small_paragraphs_tb = pd.DataFrame(columns=paragraphs_tb.columns)
        for p in paragraphs_title:
            paragraphs_title_pos.append([p.start(), p.end()])
        par_n = 1
        for i in range(0, len(paragraphs_title_pos)):
            first_ind = paragraphs_title_pos[i][0]
            if i == len(paragraphs_title_pos)-1:
                par = md_text[first_ind:]
            else:
                end_ind = paragraphs_title_pos[i+1][0]
                par = md_text[first_ind:end_ind]
            
            if len(par) > PAR_LEN_T:
                small_paragraphs_tb.loc[len(small_paragraphs_tb)] = [ref_id+par_n, title, author, par_n, par.replace("\n\n", "\n"), topic]
                par_n += 1
 
        if len(small_paragraphs_tb) < N_PAR_T:
            # Split markdown according to ** symbols
            mdf = open(f"{markdown_dir}{f[:-3]}md", "r")
            md_text = mdf.read()
            mdf.close()
            small_paragraphs_tb = check_split_strategy(md_text, ref_id, title, author, topic, paragraphs_tb.columns, PAR_LEN_T, N_PAR_T)
        else:
            print("Successfull split in paragraphs!")

        print(f"No. paragraphs: {len(small_paragraphs_tb)}, Unique content: {len(np.unique(small_paragraphs_tb["Content"]))}")
        if len(np.unique(small_paragraphs_tb["Content"])) < len(small_paragraphs_tb):
            break
    
    else:
        print(f"No paragraph pattern matches!")
        small_paragraphs_tb = check_split_strategy(md_text, ref_id, title, author, topic, paragraphs_tb.columns, PAR_LEN_T, N_PAR_T)
        print(f"No. paragraphs: {len(small_paragraphs_tb)}, Unique content: {len(np.unique(small_paragraphs_tb["Content"]))}")
        if len(np.unique(small_paragraphs_tb["Content"])) < len(small_paragraphs_tb):
            break

    paragraphs_tb = pd.concat([paragraphs_tb, small_paragraphs_tb], ignore_index=True)
    ref_id += 1
    print("=====================================================================================================")

if len(paragraphs_tb) > 0:
    paragraphs_tb.to_csv(f"../data/paragraphs_tb.csv", index=False, encoding="utf-8")

print(f"Excluded studies: {excluded_studies}")
print(f"Processed studies: {study_counter}")
