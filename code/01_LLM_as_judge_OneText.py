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

subdir = "v3/" # select dir 
data_path = f"data/{subdir}/" 

# ####################### Semantic similarity ###########################################
# for a in approaches_df.keys():
#     eval_df = Dataset.from_pandas(approaches_df[a], 
#                                   descriptors=[SemanticSimilarity(columns=['Response', 'Reference'], alias="CosineSimilarity"), BERTScore(columns=['Response', 'Reference'], alias="BertScore")])
#     res = eval_df.as_dataframe()
#     res.to_csv(f"SemanticSimilarity_{a}.csv", index=False)

############################## LLM-based Eval ##########################################
complete_df = pd.read_csv(f"{data_path}recommendation_comparison_v3.csv")

# LLM Judges
load_dotenv()
OPENAI_APIKEY = os.getenv("OPENAI_APIKEY")
CLAUDE_APIKEY = os.getenv("CLAUDE_APIKEY")
openai_options = OpenAIOptions(api_key=OPENAI_APIKEY)
claude_options = AnthropicOptions(api_key=CLAUDE_APIKEY)

# One dict of criteria 
criteria_task_prompt = {}
for c in step_2_criteria.keys():
    task_prompt = task.format(criterion=f"{c}\nDescription: {step_2_criteria[c]}", reference="{Reference}")
    criteria_task_prompt[c] = task_prompt

for c in step_3_criteria.keys():
    task_prompt = task.format(criterion=f"{c}\nDescription: {step_3_criteria[c]}", reference="{Reference}")
    criteria_task_prompt[c] = task_prompt
print(criteria_task_prompt.keys())

REQUESTS_PER_MINUTE = 2
SLEEP_TIME = 60 / REQUESTS_PER_MINUTE
f = open("debug.txt", "w")

# Set model
model = "gpt"
res_path = f"results_{model}_OneText/{subdir}"

for a in approaches:
    approaches_df = complete_df.loc[:, ["Case_Number", a, "Expert"]]
    approaches_df = approaches_df.rename(columns={a:"Response", "Expert":"Reference"})
    approaches_df = approaches_df[0:30]
    
    f.write(f"Approach: {a}\nCriteria:\n")
    for c in list(criteria_task_prompt.keys()):
        # Choose LLM and build descriptor
        f.write(f"- {c}\n{criteria_task_prompt[c]}\n--------------------\n")
        # set model
        openai_descriptor = LLMEval("Response", 
                                    provider="openai", 
                                    model="gpt-4o", 
                                    additional_columns={"Reference":"Reference"}, 
                                    template=build_template(system_prompt, criteria_task_prompt[c]))
        # claude_descriptor = LLMEval("Response", 
        #                              provider="anthropic", 
        #                              model="claude-sonnet-4-20250514", 
        #                              additional_columns={"Reference":"Reference"},
        #                              template=build_template(system_prompt, criteria_task_prompt[c]))

        llm_eval_df = Dataset.from_pandas(approaches_df, 
                                            descriptors=[openai_descriptor], options=[openai_options])
        eval_df = llm_eval_df.as_dataframe()
        eval_df.to_csv(f"{res_path}eval_{a}_{c}.csv", index=False)

        del llm_eval_df
        del openai_descriptor

        # time.sleep(SLEEP_TIME)
    f.write("------------------------------------------------------")

f.close()