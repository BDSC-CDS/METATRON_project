import pandas as pd
import json
from utils_metrics import *
from utils_variables import *

ner_path = "../data/NER/"
metrics = ["LCS","TSR","Levenshtein","Exact_Match_P"]
res_path = "results_ner/"

### Step 2 eval
for a in approaches:
    with open(f"{ner_path}entities_{a}.json", "r") as f:
        entity_dict = json.load(f)
    ner_data = pd.read_csv(f"{ner_path}entities_df_{a}.csv")

    ner_step2_oder_res = pd.DataFrame(columns=["Case_N", "LCS", "One_Hot_Encoding_DTW"])
    ner_step2_content_res = pd.DataFrame(columns=["Case_N", "Label", "Metric", "Value"])

    cases = list(set(ner_data["Case_N"]))
    labels = list(set(ner_data["Entity_Label"]))
    for c in cases:
        # Step order evaluation:
        ner_data_case = ner_data[ner_data["Case_N"] == c]
        text_list = ner_data_case.loc[ner_data_case["Source"]=="Text", "Entity_Label"].tolist()
        ref_text_list = ner_data_case.loc[ner_data_case["Source"]=="Ref_Text", "Entity_Label"].tolist()

        text_order_list = [text_list[i] for i in range(len(text_list)) if i==0 or text_list[i] != text_list[i-1]]
        ref_text_order_list = [ref_text_list[i] for i in range(len(ref_text_list)) if i==0 or ref_text_list[i] != ref_text_list[i-1]]

        # Step order evaluation: compute the sequence alignment score (LCS-based) 
        lcs = choose_fun("LCS", ref_text_order_list, text_order_list)
        
        # Step order evaluation: compute DTW between the sequences after one-hot encoding transformation
        # Repeat for each label 
        dtw_values = []
        for l in labels:
            dtw_value = choose_fun("One_Hot_Encoding_DTW", ref_text_order_list, text_order_list, l)
            dtw_values.append(dtw_value)
        ner_step2_oder_res.loc[len(ner_step2_oder_res)] = [c, lcs, np.mean(dtw_values)]

        # Content evaluation
        # for l in step2_labels:
        #     text_list = entity_dict[str(c)]["Text"][l]
        #     ref_text_list = entity_dict[str(c)]["Ref_Text"][l]

        #     for m in metrics[1:]:
        #         m_value = choose_fun(m, ref_text_list, text_list)
        #         ner_step2_content_res.loc[len(ner_step2_content_res)] = [c, l, m, m_value]
    
    ner_step2_oder_res.to_csv(f"{res_path}NER_order_metrics_Step2_{a}.csv", index=False)
    # ner_step2_content_res.to_csv(f"{res_path}NER_semantic_metric_Step2_{a}.csv", index=False)

'''
### Step 3 eval
for a in approaches:
    with open(f"{ner_path}entities_{a}_Step3.json", "r") as f:
        entity_dict = json.load(f)
    ner_data = pd.read_csv(f"{ner_path}entities_df_{a}_Step3.csv")

    ner_step3_content_res = pd.DataFrame(columns=["Case_N", "Label", "Metric", "Value"])
    
    cases = set(ner_data["Case_N"])
    for c in cases:
        # Content evaluation
        for l in step3_labels:
            text_list = entity_dict[str(c)]["Text"][l]
            ref_text_list = entity_dict[str(c)]["Ref_Text"][l]

            for m in metrics[1:]:
                m_value = choose_fun(m, ref_text_list, text_list)
                ner_step3_content_res.loc[len(ner_step3_content_res)] = [c, l, m, m_value]


    ner_step3_content_res.to_csv(f"{res_path}NER_semantic_metric_Step3_{a}.csv", index=False)
'''