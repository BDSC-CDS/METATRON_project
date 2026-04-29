import pandas as pd
import json
import pickle
from utils_metrics import *
from utils_variables import *

data_path = "../data/Taxonomy/"
metrics = ["TSR","Exact_Match_P"]
res_path = "results_taxonomy/"


version = "v5"
it = "iteration_0"
with open(f"{data_path}taxonomy_dict_Expert_first_snapshot.pickle", "rb") as f:
        reference_dict = pickle.load(f)
# c = 1
# for d in reference_dict[c].keys():
#     print("Domain ", d)
#     for l in reference_dict[c][d].keys():
#         print(f"- {l}, {reference_dict[c][d][l]}")

# For each cas...First, compute the similarity metrics between each entity in each domain
'''
excluded_entities = {a:{d:{c: [] for c in reference_dict.keys()} for d in reference_dict[1].keys()} for a in approaches}
for a in approaches:
    print(f"Approach: {a}")
    with open(f"{data_path}taxonomy_dict_{a}_{version}_{it}.pickle", "rb") as f:
        taxonomy_dict = pickle.load(f)

    content_res = pd.DataFrame(columns=["Case_N", "Domain", "Label", "Model_List", "Expert_List", "Metric", "Value"])

    for c in taxonomy_dict.keys():
        # Content evaluation
        if reference_dict[c] != None:
            for cat in taxonomy_dict[c].keys(): # domain 
                for l in taxonomy_dict[c][cat].keys(): # entities 
                    try:
                        text_list = taxonomy_dict[c][cat][l]["value"]
                        ref_text_list = reference_dict[c][cat][l]["value"]
                        # print(f"Cat {cat}; Label {l}: {text_list} vs {ref_text_list}")
                        # if ("Not mentioned" in text_list or "NA" in text_list) and ("Not mentioned" in ref_text_list or "NA" in ref_text_list):
                        if ("Not mentioned" in text_list or "NA" in text_list or "Missing" in text_list or "Absent" in text_list) or ("Not mentioned" in ref_text_list or "NA" in ref_text_list or "Missing" in ref_text_list or "Absent" in ref_text_list):
                            excluded_entities[a][cat][c].append(l) 
                        else: 
                            for m in metrics:
                                m_value = choose_fun(m, ref_text_list, text_list)
                                content_res.loc[len(content_res)] = [c, cat, l, text_list, ref_text_list, m, m_value]
                    except Exception as e:
                        print(f"Cat {cat}; Label {l}: error in key identification!")

    content_res.to_csv(f"{res_path}taxonomy_comparison_{a}_excluding.csv", index=False)

print(excluded_entities)
'''

domains = list(reference_dict[1].keys())
# For each case...Second, compute the mean across domains 
for a in approaches:
    mean_semantic_score = pd.DataFrame(columns=["Case_N","Domain","Metric","Mean","STD"])
    semantic_score_data = pd.read_csv(f"{res_path}taxonomy_comparison_{a}_excluding.csv")
    for c in reference_dict.keys():
          if reference_dict[c] != None:
            for d in domains:
                    d_data_tsr = semantic_score_data.loc[(semantic_score_data["Case_N"]==c) & (semantic_score_data["Domain"]==d) & (semantic_score_data["Metric"]=="TSR"),"Value"].values
                    d_data_exact_match = semantic_score_data.loc[(semantic_score_data["Case_N"]==c) & (semantic_score_data["Domain"]==d) & (semantic_score_data["Metric"]=="Exact_Match_P"),"Value"].values
                    values_tsr = [round(np.mean(d_data_tsr),3), round(np.std(d_data_tsr),3)]
                    values_em = [round(np.mean(d_data_exact_match),3), round(np.std(d_data_exact_match),3)]
                    row_tsr = [c, d, "TSR"] + values_tsr
                    row_em = [c, d, "Exact_Match_P"] + values_em
                    mean_semantic_score.loc[len(mean_semantic_score)] = row_tsr
                    mean_semantic_score.loc[len(mean_semantic_score)] = row_em
    mean_semantic_score.to_csv(f"{res_path}mean_semantic_metric_across_entities_{a}_excluding.csv",index=False)