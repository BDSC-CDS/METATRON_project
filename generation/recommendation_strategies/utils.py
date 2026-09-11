import os 
import re 
import pandas as pd 

def find_files(case_n: int, sr_descriptions_dir = "../structured_information_files/"):
    list_files = os.listdir(sr_descriptions_dir)
    list_files = [f for f in list_files if f"_{case_n}_" in f]
    if len(list_files) == 4:
        files_to_combine = []
        for i in range(1,5):
            step_file = [f for f in list_files if f"Step{i}" in f] 
            if len(step_file) == 1:
                files_to_combine.append(step_file[0])
        if len(files_to_combine) == 4:
            return files_to_combine
        else:
            return []
    else:
        return []

def combine_files(files_to_combine: list, sr_descriptions_dir = "../structured_information_files/"):
    complete_descr = ""
    for sf in files_to_combine:
        fr = open(f"{sr_descriptions_dir}{sf}", "r")
        fr_content = fr.read()
        if "Step1" in sf:
            complete_descr += "\n1. Primary Tumor\n"
        elif "Step2" in sf:
            complete_descr += "\n2. Liver Metastases\n"
        elif "Step3" in sf:
            complete_descr += "\n3. Extrahepatic Metastases\n"
        elif "Step4" in sf:
            complete_descr += "\n4. Anatomical feasibility for surgery\n"
        complete_descr += fr_content
        fr.close()
    return complete_descr

def clean_answer(answer: str):
    return re.sub(r"(\*\*|\*|__|_)", "", answer)
