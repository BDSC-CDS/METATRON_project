import pandas as pd 
import pickle
from utils_rag import *
from utils import *

data_path = "../data/"
prompt_dir = 'prompt/'

# Read clinical cases 
cases = pd.read_csv(data_path + "HPB_cases.csv")
print(f"No. cases: {len(cases)}, No. categories: {len(set(cases["Category"]))}")
categories = list(set(cases["Category"]))

# References data 
paragraphs_tb = pd.read_csv("data/paragraphs_tb_final.csv")

''' 
# BASIC APPROACH: No filtering
# Approach 2: Prompt for literature retrieval
f = open(prompt_dir + "rag_prompt_Approach_2.txt", "r")
rag_prompt_2 = f.read()
f.close()
# Approach 4: Prompt for literature retrieval
f = open(prompt_dir + "rag_prompt_Approach_4.txt", "r")
rag_prompt_4 = f.read()
f.close()

papers = list(set(paragraphs_tb["Title"]))
topics = list(set(paragraphs_tb["Topic"]))
paper_usage = pd.DataFrame(columns=["N_usage_Approach2", "P_usage_Approach2", "N_usage_Approach4", "P_usage_Approach4"], index=papers)
topic_usage = pd.DataFrame(columns=["N_usage_Approach2", "P_usage_Approach2", "N_usage_Approach4", "P_usage_Approach4"], index=topics)

paper_dict_2, topic_dict_2 = {p:[] for p in papers}, {t:[] for t in topics}
paper_dict_4, topic_dict_4 = {p:[] for p in papers}, {t:[] for t in topics}

for c in range(0,len(cases)):
    case_content = cases.loc[c,"Content"]
    case_n = cases.loc[c,"Case_Number"]

    # Approach 2
    rag_prompt = rag_prompt_2.format(clinical_case = case_content)
    refs = retrieval_information(rag_prompt, filter="")
    for r in refs:
        if case_n not in paper_dict_2[r["Title"]]:
            paper_dict_2[r["Title"]].append(case_n)
        if case_n not in topic_dict_2[r["Topic"]]:
            topic_dict_2[r["Topic"]].append(case_n)
    
    # Approach 4
    # Read the structured description
    str_files = find_files(case_n)
    if len(str_files) > 0:
        str_content = combine_files(str_files)
        rag_prompt = rag_prompt_4.format(clinical_case = case_content, str_analysis = str_content)
        refs = retrieval_information(rag_prompt, filter="")
        for r in refs:
            if case_n not in paper_dict_4[r["Title"]]:
                paper_dict_4[r["Title"]].append(case_n)
            if case_n not in topic_dict_4[r["Topic"]]:
                topic_dict_4[r["Topic"]].append(case_n)
    else:
        print(f"Issues in finding the structured files for case: {case_n}")

for p in paper_dict_2.keys():
    paper_usage.loc[p, "N_usage_Approach2"] = len(paper_dict_2[p])
    paper_usage.loc[p, "P_usage_Approach2"] = round((len(paper_dict_2[p])/len(cases))*100,2)
for p in paper_dict_4.keys():
    paper_usage.loc[p, "N_usage_Approach4"] = len(paper_dict_4[p])
    paper_usage.loc[p, "P_usage_Approach4"] = round((len(paper_dict_4[p])/len(cases))*100,2)

for t in topic_dict_2.keys():
    topic_usage.loc[t, "N_usage_Approach2"] = len(topic_dict_2[t])
    topic_usage.loc[t, "P_usage_Approach2"] = round((len(topic_dict_2[t])/len(cases))*100,2)
for t in topic_dict_4.keys():
    topic_usage.loc[t, "N_usage_Approach4"] = len(topic_dict_4[t])
    topic_usage.loc[t, "P_usage_Approach4"] = round((len(topic_dict_4[t])/len(cases))*100,2)

paper_usage.to_csv("results/paper_usage.csv")
topic_usage.to_csv("results/topic_usage.csv")

cases_topics_usage = pd.DataFrame(columns=list(topic_dict_2.keys()),index=cases["Case_Number"].to_list())
for c in cases["Case_Number"].to_list():
    topics = []
    for t in topic_dict_2.keys():
        if c in topic_dict_2[t]:
            cases_topics_usage.loc[c, t] = 1
cases_topics_usage.to_csv("results/cases_topics_usage_Approach2.csv")

cases_topics_usage = pd.DataFrame(columns=list(topic_dict_4.keys()),index=cases["Case_Number"].to_list())
for c in cases["Case_Number"].to_list():
    topics = []
    for t in topic_dict_4.keys():
        if c in topic_dict_4[t]:
            cases_topics_usage.loc[c, t] = 1
cases_topics_usage.to_csv("results/cases_topics_usage_Approach4.csv")
'''
###########################################################################
# Filtering Approach 1: No setting the number of refs for each topic 
# N_PT= 5
# N_LM= 20
# N_EhM= 5
# Prompt for literature retrieval
f = open(prompt_dir + "rag_prompt_Approach_2.txt", "r")
rag_prompt_template = f.read()

topics = list(set(paragraphs_tb["Topic"]))
papers = list(set(paragraphs_tb["Title"]))

paper_usage = pd.DataFrame(columns=["N_usage", "P_usage"], index=papers)
topic_usage = pd.DataFrame(columns=["N_usage", "P_usage"], index=topics)

paper_dict = {p:[] for p in papers}
topic_dict = {t:[] for t in topics}

for c in range(0, len(cases)):
    case_n = cases.loc[c,"Case_Number"]
    case_content = cases.loc[c,"Content"]
    filter = f"topic in {implement_filter(case_n)}"
    rag_prompt = rag_prompt_template.format(clinical_case = case_content)
    refs = retrieval_information(rag_prompt, filter, 30)
    for r in refs:
        if case_n not in paper_dict[r["Title"]]:
            paper_dict[r["Title"]].append(case_n)
        if case_n not in topic_dict[r["Topic"]]:
            topic_dict[r["Topic"]].append(case_n)

for p in paper_dict.keys():
    paper_usage.loc[p, "N_usage"] = len(paper_dict[p])
    paper_usage.loc[p, "P_usage"] = round((len(paper_dict[p])/len(cases))*100,2)

for t in topic_dict.keys():
    topic_usage.loc[t, "N_usage"] = len(topic_dict[t])
    topic_usage.loc[t, "P_usage"] = round((len(topic_dict[t])/len(cases))*100,2)

paper_usage.to_csv("results/paper_usage_filtering_str1.csv")
topic_usage.to_csv("results/topic_usage_filtering_str1.csv")