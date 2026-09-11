import argparse
import json
import os
from pathlib import Path

import pandas as pd

from models import *
from utils_call_models import *
from utils_parse_model_answer import *
from utils_workflow_steps import *


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WORKFLOW_PATH = SCRIPT_DIR / "workflow/workflow_information_extraction.json"


def main(
    cases,
    structured_information_path,
    output_dir,
    workflow_path=DEFAULT_WORKFLOW_PATH,
):
    structured_information_path = Path(structured_information_path)
    output_dir = Path(output_dir)
    structured_information_path.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(workflow_path, "r") as f:
        workflow_dict = json.load(f)

    cases = pd.read_csv(cases, sep=";", encoding="utf-8")
    print(f"Number of cases: {len(cases)}; columns: {cases.columns}")

    steps = ["Step1", "Step2", "Step3", "Step4"]
    condition_prompt_log = {"Failed": 0, "Simple_Prompt": 0, "Structured_Prompt": 0}
    sie_prompt_log = {"Failed": 0, "Simple_Prompt": 0, "Structured_Prompt": 0}
    conditional_steps_results = pd.DataFrame(
        columns=["Case", "Step", "Substep", "Condition", "Verified"]
    )
    features_tb_items = {}
    imaging_response = None

    with open(output_dir / "debug_file.txt", "w", encoding="utf-8") as debug_file:
        for case in range(0, len(cases)):
            case_n = cases.loc[case, "Case_Number"]
            case_content = cases.loc[case, "Content"]
            debug_file.write(f"Case: {case}\n")
            print(f"Elaborating Case {case_n}\nIt starts as: {case_content[0:50]}")

            for step in steps:
                structured_information_file = open(
                    os.path.join(
                        structured_information_path,
                        f"structured_information_{case_n}_{step}.txt",
                    ),
                    "w",
                )
                try:
                    print(f"STEP {step}")
                    debug_file.write(f"STEP {step}\n")
                    for item in workflow_dict[step]["Items"]:
                        item_name = list(item.keys())[0]
                        item_content = item[item_name]
                        structured_information_file.write(
                            f"\nInformation concerning {item_name}\n"
                        )

                        if "Description" in item_content:
                            structured_information_file.write(
                                f"Description: {item_content['Description']}\n"
                            )

                        if step == "Step3":
                            execute_step(
                                case_n,
                                case_content,
                                item_content,
                                item_name,
                                sie_prompt_log,
                                condition_prompt_log,
                                debug_file,
                                conditional_steps_results,
                                features_tb_items,
                                step,
                                structured_information_file,
                                imaging_response,
                            )
                        else:
                            execute_step(
                                case_n,
                                case_content,
                                item_content,
                                item_name,
                                sie_prompt_log,
                                condition_prompt_log,
                                debug_file,
                                conditional_steps_results,
                                features_tb_items,
                                step,
                                structured_information_file,
                            )

                        if step == "Step2" and item_name == "ImagingExams":
                            ct_results = features_tb_items["CT_Scan_Performed"]
                            thoraco_ct = ct_results.loc[
                                len(ct_results) - 1, "Thoraco_CT_Scan_Performed"
                            ]
                            if thoraco_ct != "Missing":
                                if not isinstance(thoraco_ct, np.bool):
                                    thoraco_ct = np.bool(thoraco_ct)
                            abdomen_ct = ct_results.loc[
                                len(ct_results) - 1, "Abdomen_CT_Scan_Performed"
                            ]
                            if abdomen_ct != "Missing":
                                if not isinstance(abdomen_ct, np.bool):
                                    abdomen_ct = np.bool(abdomen_ct)
                            imaging_response = [thoraco_ct, abdomen_ct]
                finally:
                    structured_information_file.close()

    print("Condition")
    print(conditional_steps_results)

    print("-----------------------------\nIE")
    for item in features_tb_items:
        if features_tb_items[item] is not None:
            print(f"features_tb: {item}\n{features_tb_items[item]}")

    conditional_steps_results.to_csv(
        os.path.join(output_dir, "conditional_steps_results.csv"),
        index=False,
        sep=";",
    )

    for item in features_tb_items:
        if features_tb_items[item] is not None:
            features_tb_items[item].to_csv(
                os.path.join(output_dir, f"sie_{item}.csv"),
                index=False,
                sep=";",
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Execute the structured information extraction workflow."
    )
    parser.add_argument(
        "--cases",
        required=True,
        help="Complete path to the cases CSV.",
    )
    parser.add_argument(
        "--structured-information-path",
        required=True,
        help="Complete path for the generated structured-information files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for workflow results and debug output.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(**vars(parse_args()))
