import pandas as pd
import bedrock_inference
import os
import boto3
import instructor  # library for get reliable JSON from any LLM
import pickle
import ast
from openai import OpenAI
from dotenv import load_dotenv, set_key
from utils_prompt import *
from EvaluationOutput import *
# from BedrockCaller import *
from utils_call_bedrock_models import *
from utils_variables import *
from query_gpt import query_gpt

import random
flag = random.uniform(0, 1)

########### JUDGES ##################
# JUdges from bedrock
# Once you have temporary credentials, you can use them to create a Bedrock client
# Rerun the script to use updated creds
load_dotenv(cred_path + "temporary_cred.env") # load temporary credentials
access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
access_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
session_token = os.getenv("AWS_SESSION_TOKEN")

# Bedrock client
bedrock_client = boto3.client("bedrock-runtime",
                aws_access_key_id=access_key_id,
                aws_secret_access_key=access_secret_key,
                aws_session_token=session_token,
                region_name="us-west-2"
)

# Then, create client using instructor
client = instructor.from_bedrock(bedrock_client)
# Set model
models_id_list = ["us.anthropic.claude-sonnet-4-6", "us.anthropic.claude-sonnet-4-20250514-v1:0", "us.meta.llama3-3-70b-instruct-v1:0"]
model = models_id_list[2]
model_name = "llama"

# Or use gpt model
# load_dotenv()
# llm_api_key=os.getenv("OPENAI_APIKEY")
# client = OpenAI(api_key=llm_api_key)
# # Set model
# model = "gpt-4o"
# model_name = "gpt"

######### JUDGES EVAL #############
strategy = "OneText_OnePrompt"
version = "v5"
iteration = "iteration_2"

data_path = '../data/'
res_path_model = f"results_{model_name}_{strategy}/{version}_{iteration}/"
res_path = "results"

approaches_df_complete = pd.read_csv(f"{data_path}{version}/recommendation_comparison_{version}_{iteration}.csv")
all_cases = approaches_df_complete["Case_Number"].to_list()
print(f"Case included: {all_cases}")

# For small model: structural errors 
# errors = pd.read_csv(f"{data_path}{version}/step_errors_{version}_{iteration}_small_model.csv")
# case_to_exclude = []
# for i in range(0,len(errors)):
#     case_err = errors.loc[i,'Case_Not_Correct_Structure']
#     case_err_list = ast.literal_eval(case_err)
#     if isinstance(case_err_list, list):
#         case_to_exclude += case_err_list
# unique_case_to_exclude = list(set(case_to_exclude))
# print(f"Unique case to exclude: {len(unique_case_to_exclude)}\n{unique_case_to_exclude}")

for run in range(2, 3):
    print(f"Run: {run}")
    for a in approaches[-1:]:
        f = open(f"debug/prompt_{a}.txt", "w", encoding="utf-8")
        print(f"Approach: {a}")
        f.write(f"Approach: {a}\n")

        tb = pd.DataFrame(columns=["Case_Number","Case_Content","Response","Reference","Completeness","Step_concordance","Case_tailoring","Missing_data_concordance"])
        # tb = pd.read_csv(f"{res_path_model}eval_{a}_first_snapshot_run_{run}.csv") # read existing results to keep adding to them
        # if a == "Model_Alone":
            # included_cases = tb["Case_Number"].to_list()
            # remaining_cases = [rc for rc in all_cases if rc not in included_cases]
            # print(f"Cases to be evaluated: {remaining_cases}")
            # explanation_cases = random.sample(remaining_cases, 10)
            # print(f"Cases with explanation: {explanation_cases}")
        explanation_cases = [43, 46, 48, 49, 62, 64, 65, 74, 82, 100]
        
        explanation_tb = pd.DataFrame(columns=["Case_Number","Case_Content","Response","Reference","Completeness","Step_concordance","Case_tailoring","Missing_data_concordance", "Reasoning"])
        approaches_df = approaches_df_complete.loc[:, ["Case_Number", "Content", a, "Expert"]]

        for case in range(0,len(approaches_df_complete)):
            case_n = approaches_df.loc[case,"Case_Number"].item()
            # if case_n in remaining_cases:
            if flag == 0:
                print("Expert is r1")
                r1 = approaches_df.loc[case, "Expert"]
                r2 = approaches_df.loc[case, a]
            else:
                r1 = approaches_df.loc[case, a]
                r2 = approaches_df.loc[case, "Expert"]
            u_prompt = task.format(completeness=completeness, step_concordance=step_concordance, case_tailoring=case_tailoring, missing_data_concordance=missing_data_concordance,
                                r1=r1, r2=r2)
            # prompt example
            f.write(f"Case {case}\n{u_prompt}")
            if case_n in explanation_cases and run == 0:
                answer = query_models(client, system_prompt, u_prompt, model, EvaluationOutputReasoning)
                # answer = query_gpt(system_prompt, u_prompt, model, client, EvaluationOutputReasoning)
            else:
                answer = query_models(client, system_prompt, u_prompt, model, EvaluationOutput)
                # answer = query_gpt(system_prompt, u_prompt, model, client, EvaluationOutput)                

            if answer != None:
                if isinstance(answer, dict):
                    tb.loc[len(tb)] = [approaches_df.loc[case,"Case_Number"].item(),approaches_df.loc[case,"Content"],approaches_df.loc[case,a],
                                    approaches_df.loc[case, "Expert"], answer["completeness"], answer["step_concordance"], answer["case_tailoring"], 
                                    answer["missing_data_concordance"]]
                    if case_n in explanation_cases and run == 0:
                        explanation_tb.loc[len(explanation_tb)] = [approaches_df.loc[case,"Case_Number"].item(),approaches_df.loc[case,"Content"],approaches_df.loc[case,a],
                                        approaches_df.loc[case, "Expert"], answer["completeness"], answer["step_concordance"], answer["case_tailoring"], 
                                        answer["missing_data_concordance"], answer["reasoning"]]
                else:
                    print(f"Issues with answer: {answer}, case: {case}")

        # Sort cases 
        tb = tb.sort_values("Case_Number")
        print(f"Saving...N={len(tb)}")
        tb.to_csv(f"{res_path_model}eval_{a}_run_{run}.csv", index=False)
        print(f"Explanation: Saving...N={len(explanation_tb)}")
        explanation_tb.to_csv(f"{res_path_model}eval_{a}_run_{run}_with_explanation.csv", index=False)