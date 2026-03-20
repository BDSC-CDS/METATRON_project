import pandas as pd 
import os
import pickle
from utils_variables import *

################################## Collect results ###################################
# More res dirs 
'''
model = "gpt"
res_dir_names = ["1-5", "6", "7-11", "12-21", "22-30", "30-50", "50-100"]
new_res_dir = "results/"

criteria_df_dict = {c:{a: None for a in approaches} for c in criteria}

for c in criteria:
    for a in approaches:
        criterion_df = pd.DataFrame(columns=["Case_Number","Case_Content","Response","Reference","LLMEval"])
        f_name = f"eval_{a}_{c}.csv"
        for d in res_dir_names:
            dir_name = f"results_{model}_{d}"
            f_list = os.listdir(dir_name)
            f_list_criteria = [f for f in f_list if c in f]
            if len(f_list_criteria) != len(criteria):
                print(f"Warning: not all approaches have results for criterion {c} in directory {dir_name}")
            else:
                if f_name not in f_list_criteria:
                    print(f"Warning: no results for approach {a} and criterion {c} in directory {dir_name}: {f_name}")
                else:
                    df = pd.read_csv(f"{dir_name}/{f_name}")
                    criterion_df = pd.concat([criterion_df, df], ignore_index=True)
                    criteria_df_dict[c][a] = criterion_df

for c in criteria_df_dict.keys():
    print(f"Criterion: {c}, N. df: {len(criteria_df_dict[c])}")
    for a in criteria_df_dict[c].keys():
        print(f"Approach: {a}, N. of samples: {len(criteria_df_dict[c][a])}")

with open(f"{new_res_dir}complete_results_{model}.pickle", "wb") as f:
    pickle.dump(criteria_df_dict, f)
'''

# One res dir 
model = "gpt" # Set model
strategy = "OneText" # Set strategy
version = "v3"
dir_name = f"results_{model}_{strategy}/{version}/"
new_res_dir = "results/"

criteria_df_dict = {c:{a: None for a in approaches} for c in criteria}

for c in criteria:
    for a in approaches:
        criterion_df = pd.DataFrame(columns=["Case_Number","Case_Content","Response","Reference","LLMEval"])
        f_name = f"eval_{a}_{c}.csv"
        f_list = os.listdir(dir_name)
        f_list_criteria = [f for f in f_list if c in f]
        if len(f_list_criteria) != len(criteria):
            print(f"Warning: not all approaches have results for criterion {c} in directory {dir_name}")
        else:
            if f_name not in f_list_criteria:
                print(f"Warning: no results for approach {a} and criterion {c} in directory {dir_name}: {f_name}")
            else:
                df = pd.read_csv(f"{dir_name}/{f_name}")
                criterion_df = pd.concat([criterion_df, df], ignore_index=True)
                criteria_df_dict[c][a] = criterion_df

for c in criteria_df_dict.keys():
    print(f"Criterion: {c}, N. df: {len(criteria_df_dict[c])}")
    for a in criteria_df_dict[c].keys():
        print(f"Approach: {a}, N. of samples: {len(criteria_df_dict[c][a])}")

with open(f"{new_res_dir}complete_results_{model}_{strategy}_{version}.pickle", "wb") as f:
    pickle.dump(criteria_df_dict, f)


