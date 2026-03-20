import pandas as pd
import bedrock_inference
import os
import boto3
import instructor  # library for get reliable JSON from any LLM
import pickle
from dotenv import load_dotenv, set_key
from utils_prompt import *
from EvaluationOutput import *
# from BedrockCaller import *
from utils_call_bedrock_models import *
from utils_variables import *

'''
    Create bedrock client 
'''
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


'''
    Use Models
'''
# LLM list 
models_id_list = ["us.anthropic.claude-sonnet-4-20250514-v1:0"]
model = models_id_list[0]
model_name = "claude"

strategy = "OneText"

data_path = '../data/'
res_path_model = f"../results_{model}_{strategy}/"
res_path = "results"

# Criteria-based prompt
criteria_prompt = {c:"" for c in list(step_2_criteria) + list(step_3_criteria)}

# Generate and save evaluations
evaluation_dict = {}

for s in ["step2", "step3"]:
    print(f"Evaluating {s}...\n")

    if s == "step2":
        criteria = step_2_criteria
        criteria_task_prompt = step_2_criteria
    elif s == "step3":
        criteria = step_3_criteria
        criteria_task_prompt = step_3_criteria

    for c in criteria_task_prompt.keys():
        print(f"Criterion: {c}")
        criteria_dict = {}

        for a in approaches:
            f = open(f"debug/prompt_{s}_{a}_{c}.txt", "w", encoding="utf-8")
            print(f"Approach: {a}")
            f.write(f"Approach: {a}\n")
            f.write(f"Criterion: {c}\n")
            tb = pd.DataFrame(columns=["Case_Number","Case_Content","Response","Reference","LLMEval"])

            approaches_df = pd.read_csv(f"{data_path}{s}_{a}.csv")
            approaches_df = approaches_df[0:30]

            for case in range(len(approaches_df)):
                u_prompt = task.format(criterion=criteria_task_prompt[c], reference=approaches_df.loc[case, "Reference"],
                                       junior=approaches_df.loc[case,"Response"])
                # prompt example
                f.write(f"Case {case}\n{u_prompt}")
                answer = query_models(client, system_prompt, u_prompt, model, EvaluationOutput)
                #print(answer, type(answer))
                if answer != None:
                    if isinstance(answer, dict):
                        if "LLMEval" in answer.keys():
                            tb.loc[len(tb)] = [approaches_df.loc[case,"Case_Number"].item(),approaches_df.loc[case,"Case_Content"],approaches_df.loc[case,"Response"],
                                            approaches_df.loc[case, "Reference"], answer["LLMEval"]]
                    else:
                        print(f"Issues with answer: {answer}, case: {case}")
            print(f"Saving...N={len(tb)}")
            tb.to_csv(f"{res_path_model}eval_{a}_{c}.csv", index=False)
            criteria_dict[a] = tb
        evaluation_dict[c] = criteria_dict

with open(f"{res_path}complete_results_{model_name}_{strategy}.pickle", "wb") as f:
    pickle.dump(evaluation_dict, f)



