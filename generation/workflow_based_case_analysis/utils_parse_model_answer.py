import json
import pandas as pd
import numpy as np 
import re

def extract_answer(model_answer: str):
    start_strings = ["assistantfinal", "final answer", "let's craft"]
    for s in start_strings:
        index = model_answer.find(s)
        if index > -1:
            answer = model_answer[index + len(s):]
            return answer
    print("No start string found!")
    return model_answer

def process_structured_output(model_answer: str, item_name: str):
    json_start = model_answer.find("{")
    json_stop = model_answer.rfind("}")
    if json_start > -1 and json_stop > -1:
        json_str = model_answer[json_start:json_stop+1]
        try:
            json_dict = json.loads(json_str)
            return json_dict
        except:       
            return json_str
    else:
        return model_answer

def build_features_table(case_n: int, json_dict: dict, item_name: str, features_tb: pd.DataFrame, debug_file, str_file):
    items = json_dict[item_name]["Items"]
    if len(items) > 0:
        columns_name = ["Case_ID"]+[list(el.keys())[0] for el in items]
        values_ = [list(el.values())[0] for el in items]
        if str_file is not None:
            for col_name, val in zip(columns_name[1:], values_):
                str_file.write(f"- {col_name}: {val}\n")
        if features_tb is None:
            features_tb = pd.DataFrame(data=[[case_n] + values_], columns=columns_name)
        else:
            new_row = pd.DataFrame(data=[[case_n] + values_], columns=columns_name)
            features_tb = pd.concat([features_tb, new_row], ignore_index=True)
        return features_tb
    else:
        debug_file.write(f"SIE Step: Zero-len items {item_name}: {items}\n")
        return features_tb

def extract_condition_output(model_answer: str, imaging_response=None):
    matches_pos, matches_neg, matches_miss = [], [], []
    response = None
    pattern_pos = r'yes'
    pattern_neg = r'no'
    pattern_missing = r"missing"
    flag = False
    matches_pos = re.findall(pattern_pos, model_answer, flags=re.IGNORECASE)
    matches_neg = re.findall(pattern_neg, model_answer, flags=re.IGNORECASE)
    matches_miss = re.findall(pattern_missing, model_answer, flags=re.IGNORECASE)
    if len(matches_pos) > 0:
        response = True
    elif len(matches_neg) > 0:
        response = False
    elif len(matches_miss) > 0:
        if imaging_response is not None:
            if isinstance(imaging_response, np.bool):
                if imaging_response:
                    flag = True
                    response = "Assumed False"
                else:
                    response = "Missing"
            else:
                response = "Missing"
        else:
            response = "Missing"
    print(f"Imaging Response: {imaging_response}, Response: {response}")
    if flag and response != "Assumed False":
        raise ValueError("The condition of Assumed False has not worked!")
    return response, matches_pos, matches_neg, matches_miss

