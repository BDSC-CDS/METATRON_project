import ast
import pandas as pd
import numpy as np

def extract_structured_output(text: str):
    try:
        dict_ = ast.literal_eval(text)
        return dict_
    except:
        print("No structured output provided!")
        return None
    
def extract_items(dict_: dict, keys_list: list):
    items = []
    for k in keys_list:
        try:
            if not(pd.isnull(dict_[k])):
                if dict_[k] != "":
                    items.append(dict_[k])
                else:
                    print(f"Item for {k} is empty string")
            else:
                print(f"Item for {k} is null")
        except:
            print(f"{k} not found!")
    return items