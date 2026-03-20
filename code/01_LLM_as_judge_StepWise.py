from dotenv import load_dotenv
import os   
import pandas as pd 
import numpy as np
from evidently.descriptors import CorrectnessLLMEval, LLMEval
from evidently.llm.templates import BinaryClassificationPromptTemplate
from evidently import Dataset
from evidently.descriptors import TextLength, IncludesWords, SemanticSimilarity, BERTScore, SentenceCount
from evidently.llm.options import OpenAIOptions, AnthropicOptions
import time
import warnings
warnings.filterwarnings("ignore")
from utils_prompt import *
from utils_variables import *

# Functions
def build_template(system_prompt, task):
    correctness_criterion = BinaryClassificationPromptTemplate(
    pre_messages=[("system", system_prompt)],
    criteria=task,
    target_category="SATISFIED",
    non_target_category="NOT SATISFIED",
    uncertainty="unknown",
    include_reasoning=False,
    include_scores=False
)
    return correctness_criterion

data_path = "data/"

# ####################### Semantic similarity ###########################################
# for a in approaches_df.keys():
#     eval_df = Dataset.from_pandas(approaches_df[a], 
#                                   descriptors=[SemanticSimilarity(columns=['Response', 'Reference'], alias="CosineSimilarity"), BERTScore(columns=['Response', 'Reference'], alias="BertScore")])
#     res = eval_df.as_dataframe()
#     res.to_csv(f"SemanticSimilarity_{a}.csv", index=False)

############################## LLM-based Eval ##########################################

# LLM Judges
load_dotenv()
OPENAI_APIKEY = os.getenv("OPENAI_APIKEY")
CLAUDE_APIKEY = os.getenv("CLAUDE_APIKEY")
openai_options = OpenAIOptions(api_key=OPENAI_APIKEY)
claude_options = AnthropicOptions(api_key=CLAUDE_APIKEY)

# Evaluate step 1 and 2 using Completeness, Step-by-step concordance, and Case-tailoring concordance criteria
step_12_criteria_task_prompt = {}
for c in step_12_criteria.keys():
    task_prompt = task.format(criterion=f"{c}\nDescription: {step_12_criteria[c]}", reference="{Reference}")
    step_12_criteria_task_prompt[c] = task_prompt
# print(f"Step 1 and 2 criteria task prompts: {step_12_criteria_task_prompt}")

# Evaluate step 3 using Missing data concordance criterion
step_3_criteria_task_prompt = {}
for c in step_3_criteria.keys():
    task_prompt = task.format(criterion=f"{c}\nDescription: {step_3_criteria[c]}", reference="{Reference}")
    step_3_criteria_task_prompt[c] = task_prompt
# print(f"Step 3 criteria task prompts: {step_3_criteria_task_prompt}")

REQUESTS_PER_MINUTE = 2
SLEEP_TIME = 60 / REQUESTS_PER_MINUTE
f = open("debug.txt", "w")

# Set model
model = "gpt"
res_path = f"results_{model}_StepWise/"

for s in ["step12", "step3"]:
    f.write(f"Evaluating {s}...\n")

    if s == "step12":
        criteria = step_12_criteria
        criteria_task_prompt = step_12_criteria_task_prompt
    elif s == "step3":
        criteria = step_3_criteria
        criteria_task_prompt = step_3_criteria_task_prompt
    for a in approaches:
        approaches_df = pd.read_csv(f"{data_path}{s}_{a}.csv")
        approaches_df = approaches_df.iloc[0:1]
        
        f.write(f"Approach: {a}\nCriteria:\n")
        for c in list(criteria.keys()):
            # Choose LLM and build descriptor
            f.write(f"- {c}\n{criteria_task_prompt[c]}\n--------------------\n")
            openai_descriptor = LLMEval("Response", 
                                        provider="openai", 
                                        model="gpt-4o", 
                                        additional_columns={"Reference":"Reference"}, 
                                        template=build_template(system_prompt, criteria_task_prompt[c]))
            claude_descriptor = LLMEval("Response", 
                                         provider="anthropic", 
                                         model="claude-sonnet-4-20250514", 
                                         # model="claude-sonnet-4-5-20250929",
                                         additional_columns={"Reference":"Reference"},
                                         template=build_template(system_prompt, criteria_task_prompt[c]))

            llm_eval_df = Dataset.from_pandas(approaches_df, 
                                              descriptors=[claude_descriptor]
                                              #options=[claude_options]
                                              )
            eval_df = llm_eval_df.as_dataframe()
            eval_df.to_csv(f"{res_path}eval_{a}_{c}.csv", index=False)

            del llm_eval_df
            del openai_descriptor

            # time.sleep(SLEEP_TIME)
        f.write("------------------------------------------------------")
    f.write("===================================================================================")

f.close()