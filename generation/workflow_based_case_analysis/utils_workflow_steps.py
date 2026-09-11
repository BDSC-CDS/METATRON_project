import io
from pathlib import Path
from utils_call_models import *
from utils_parse_model_answer import *

# User ans System Prompt dir
SCRIPT_DIR = Path(__file__).resolve().parent
prompt_dir = SCRIPT_DIR / "prompt"

def read_prompt_template(step: str):
    user_prompt_template = open(prompt_dir /"user_prompt_ie.txt", "r").read()
    if step != "Step4":
        question_prompt_template = open(prompt_dir /"question_prompt.txt", "r").read()
    else:
        question_prompt_template = open(prompt_dir /"question_prompt_step4.txt", "r").read()
    sys_prompt = open(prompt_dir /"system_prompt.txt", "r").read()

    # Complete Prompt (Fallback Prompt)
    complete_prompt_ie_template = open(prompt_dir /"complete_prompt_ie.txt", "r").read()
    complete_prompt_question = open(prompt_dir /"complete_prompt_question.txt", "r").read()
    return user_prompt_template, question_prompt_template, sys_prompt, complete_prompt_ie_template, complete_prompt_question

# Query model
client = OpenAI(
    api_key="empty",
    base_url=models["llm"]["url"])

def condition_step(case_content: str, substep: dict, substep_name: str, prompt_log: dict, question_prompt_template: str, sys_prompt: str, complete_prompt_question: str, str_file=None, imaging_response=None):
    if imaging_response is not None:
        if substep_name == "Extrahepatic_Brain_Involvement": # For brain involvement, not consider the previous imaging
            imaging_response = None
        elif substep_name == "Extrahepatic_Pulmonary_Involvement":
            imaging_response = imaging_response[0] # For pulmonary involvement, consider thoraco CT imaging
        elif substep_name == "Extrahepatic_Bone_Involvement":
            if isinstance(imaging_response[0], np.bool) and isinstance(imaging_response[1], np.bool):
                imaging_response = imaging_response[0] and imaging_response[1] # For bone involvement, consider both types of CT scan
            else:
                imaging_response = "Missing"
        else:
            imaging_response = imaging_response[1] # For other involvements, consider abdomen CT scan
    output = None
    pos_output, neg_output, missing_output = [], [], []
    question_prompt = question_prompt_template.format(clinical_case=case_content, question=substep[substep_name]["Condition"]["Question"])
    # print(f"Item {substep_name}, prompt:\n{question_prompt}")
    answer = query_model_structured_message(client, sys_prompt, question_prompt)
    # print(f"Answer: {answer}\n-----------------------------------------\n")
    if answer is None:
        question_prompt = complete_prompt_question.format(clinical_case=case_content, question=substep[substep_name]["Condition"]["Question"])
        answer = query_model(client, question_prompt)
        if answer is not None:
            prompt_log["Simple_Prompt"] += 1
            output = extract_condition_output(answer, imaging_response)
        else:
            prompt_log["Failed"] += 1
    else:
        prompt_log["Structured_Prompt"] += 1
        output, pos_output, neg_output, missing_output = extract_condition_output(answer, imaging_response)
    if str_file is not None:
        str_file.write(f"- {substep[substep_name]["Condition"]["Question"]} {output}\n")
    return output, pos_output, neg_output, missing_output, prompt_log



def structured_information_extraction_step(case_n: int, case_content: str, substep: dict, substep_name: str, prompt_log: dict, features_tb: pd.DataFrame, debug_file: io.TextIOWrapper, user_prompt_template: str, sys_prompt: str, complete_prompt_ie_template: str, str_file=None):
    str_dict = json.dumps(substep, indent=2)
    user_prompt = user_prompt_template.format(clinical_case=case_content, json_input=str_dict)
    # print(f"Item {substep_name}, prompt:\n{user_prompt}")
    answer = query_model_structured_message(client, sys_prompt, user_prompt)
    # print(f"Answer: {answer}\n-----------------------------------------\n")
    if answer is None:
        user_prompt = complete_prompt_ie_template.format(clinical_case=case_content, json_input=str_dict)
        answer = query_model(client, user_prompt)
        if answer is not None:
            prompt_log["Simple_Prompt"] += 1
            try:
                dict_content = process_structured_output(answer, substep_name)

                features_tb = build_features_table(case_n, dict_content, substep_name, features_tb, debug_file, str_file)
            except Exception as e:
                debug_file.write(f"SIE Step; Exception in processing item {substep_name}. Exception: {e}\nAnswer: {answer}\n")
                return answer, features_tb, prompt_log
        else:
            prompt_log["Failed"] += 1
    else:
        prompt_log["Structured_Prompt"] += 1
        try:
            dict_content = process_structured_output(answer, substep_name)
            features_tb = build_features_table(case_n, dict_content, substep_name, features_tb, debug_file, str_file)
        except Exception as e:
            debug_file.write(f"SIE Step; Exception in processing item {substep_name}. Exception: {e}\nAnswer: {answer}\n")
            return answer, features_tb, prompt_log
    return dict_content, features_tb, prompt_log



def extract_condition_node_content(node: dict):
    new_node = {}
    for k in node.keys():
        if k != "Condition":
            new_node.update({k:node[k]})
    return new_node



# Workflow Execution
def execute_step(case_n: int, case_content: str, node: dict, node_name: str, sie_prompt_log: dict, condition_prompt_log: dict, debug_file: io.TextIOWrapper, conditional_steps_results: dict, features_tb_items: dict,
                 step: str, str_file=None, imaging_response=None):
    # print(f"Node name {node_name}, Content:\n{node}")
    user_prompt_template, question_prompt_template, sys_prompt, complete_prompt_ie_template, complete_prompt_question = read_prompt_template(step)
    # Condition node 
    if isinstance(node, dict):
        if "Condition" in node:
            response, positive_response, negative_response, miss_response, condition_prompt_log = condition_step(case_content, {node_name: node}, node_name, condition_prompt_log, question_prompt_template, sys_prompt, complete_prompt_question, str_file, imaging_response)
            if response is None:
                debug_file.write(f"CONDITION STEP: {node['Condition']['Question']}. Failed condition response extraction: Positive: {positive_response}, Negative: {negative_response}, Missing: {miss_response}\n")
            else:
                conditional_steps_results.loc[len(conditional_steps_results)] = [case_n, step, node_name, node["Condition"]["Question"], response]

            # If Condition is True --> extract information (execute step)
            if isinstance(response, bool) and response:
                dict_ = extract_condition_node_content(node)
                execute_step(case_n, case_content, dict_, node_name, sie_prompt_log, condition_prompt_log, debug_file, conditional_steps_results, features_tb_items, step, str_file, imaging_response)

        # No condition to verify, but check the SubItems 
        elif "Condition" not in node and "Items" in node:
            flag = False
            for item in node["Items"]:
                item_name = list(item.keys())[0]
                item_content = item[item_name]
                if "Condition" in item_content:
                    flag = True
                    break 
            if flag:
                for item in node["Items"]:
                    item_name = list(item.keys())[0]
                    item_content = item[item_name]
                    execute_step(case_n, case_content, item_content, item_name, sie_prompt_log, condition_prompt_log, debug_file, conditional_steps_results, features_tb_items, step, str_file, imaging_response)
            else:
                dict_content, features_tb, sie_prompt_log = structured_information_extraction_step(case_n, case_content, {node_name: node}, node_name, sie_prompt_log, features_tb_items.get(node_name), debug_file, user_prompt_template, sys_prompt, complete_prompt_ie_template, str_file)
                features_tb_items[node_name] = features_tb
    else:
        dict_content, features_tb, sie_prompt_log = structured_information_extraction_step(case_n, case_content, {node_name: {"Items":[{node_name: node}]}}, node_name, sie_prompt_log, features_tb_items.get(node_name), debug_file, user_prompt_template, sys_prompt, complete_prompt_ie_template, str_file)
        features_tb_items[node_name] = features_tb