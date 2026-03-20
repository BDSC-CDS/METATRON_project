import pandas as pd
from utils_variables import *

data_path = "data/v3/"

##################### Prepare dataset for step-based approach #######################################
# Load data and create one dataset for each approach, splitting by step and including reference recommendation
df = pd.read_csv(f"{data_path}recommendation_steps_v3.csv")
complete_df = pd.read_csv(f"{data_path}recommendation_comparison_v3.csv")

step12_approaches_df, step3_approaches_df = {},{}
for approach in approaches:
    step_12_df = pd.DataFrame(columns=["Case_Number", "Case_Content", "Response", "Reference"])
    step_3_df = pd.DataFrame(columns=["Case_Number", "Case_Content", "Response", "Reference"])
    for case in range(0, len(df)):
        case_content = complete_df.loc[complete_df["Case_Number"]==case+1, "Content"].item()

        model_answer = df.loc[case, f"{approach}_Step 1"] + df.loc[case, f"{approach}_Step 2"]
        ref_answer = df.loc[case, f"Expert_Step 1"] + df.loc[case, f"Expert_Step 2"]
        step_12_df.loc[len(step_12_df)] = [case+1, case_content, model_answer, ref_answer]
    
        model_answer = df.loc[case, f"{approach}_Step 3"]
        ref_answer = df.loc[case, f"Expert_Step 3"]
        step_3_df.loc[len(step_3_df)] = [case+1, case_content, model_answer, ref_answer]
    
    step12_approaches_df[approach] = step_12_df
    step3_approaches_df[approach] = step_3_df

for a in approaches:
    step12_approaches_df[a].to_csv(f"{data_path}step12_{a}.csv", index=False)
    step3_approaches_df[a].to_csv(f"{data_path}step3_{a}.csv", index=False)

# for a in approaches_df.keys():
#     print(f"Approach: {a}, N. of samples: {len(approaches_df[a])}")
# test = approaches_df["Model_Alone"]
# print(test.columns)
#
