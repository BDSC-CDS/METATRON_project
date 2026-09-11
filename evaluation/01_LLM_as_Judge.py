import argparse
import os
import random

import pandas as pd

from EvaluationOutput import EvaluationOutputReasoning
from query_gpt import query_gpt
from utils_call_bedrock_models import query_models


APPROACHES = ["Model_Alone", "Model_Studies", "Model_Workflow", "Model_Workflow_Studies"]


def main(
    n_selected_cases: int,
    model: str,
    model_env: str,
    client: object,
    data: pd.DataFrame,
) -> None:
    """Evaluate selected cases using the supplied model client and dataset."""
    if n_selected_cases < 1:
        raise ValueError("N_SELECTED_CASES must be greater than zero.")
    if n_selected_cases > len(data):
        raise ValueError("N_SELECTED_CASES cannot exceed the number of cases in data.")

    version = data.attrs.get("version", "v8")
    if ":" in model:
        model = model.replace(":", "")
    result_path = f"results_{model}/{version}/"
    os.makedirs(result_path, exist_ok=True)
    os.makedirs("debug", exist_ok=True)

    all_cases = data["Case_Number"].to_list()
    print(f"Case included: {all_cases}\nNumber: {len(all_cases)}")
    random.seed(42)
    selected_cases = random.sample(all_cases, n_selected_cases)
    print(f"Selected: {selected_cases}")

    system_prompt = open("system_prompt_eval.txt", "r", encoding="utf-8").read()
    user_prompt_template = open("user_prompt_eval.txt", "r", encoding="utf-8").read()

    for approach in APPROACHES:
        with open(f"debug/prompt_{approach}.txt", "w", encoding="utf-8") as prompt_file:
            print(f"Approach: {approach}")
            prompt_file.write(f"Approach: {approach}\n")

            explanation_table = pd.DataFrame(
                columns=[
                    "Case_Number", "Case_Content", "Response", "Reference",
                    "Completeness", "Step_concordance", "Case_tailoring",
                    "Missing_data_concordance", "MDT_Decision_Concordance", "Reasoning",
                ]
            )
            approaches_data = data.loc[:, ["Case_Number", "Content", approach, "Expert"]]

            for case in selected_cases:
                case_data = approaches_data.loc[
                    approaches_data["Case_Number"] == case
                ].iloc[0]
                user_prompt = user_prompt_template.format(
                    case=case_data["Content"],
                    reference=case_data["Expert"],
                    candidate=case_data[approach],
                )
                prompt_file.write(f"Case {case}\n{user_prompt}")

                if model_env.lower() in {"aws", "aws bedrock", "bedrock"}:
                    answer = query_models(
                        client, system_prompt, user_prompt, model, EvaluationOutputReasoning
                    )
                else:
                    answer = query_gpt(
                        system_prompt, user_prompt, model, client, EvaluationOutputReasoning
                    )

                if isinstance(answer, dict):
                    explanation_table.loc[len(explanation_table)] = [
                        case_data["Case_Number"],
                        case_data["Content"],
                        case_data[approach],
                        case_data["Expert"],
                        answer["completeness"],
                        answer["step_concordance"],
                        answer["case_tailoring"],
                        answer["missing_data_concordance"],
                        answer["mdt_decision_concordance"],
                        answer["reasoning"],
                    ]
                elif answer is not None:
                    print(
                        f"Issues with answer of approach {approach}, "
                        f"case {case} --> It is not a dict!"
                    )

        explanation_table.to_csv(
            f"{result_path}eval_{approach}_with_explanation.csv",
            index=False,
            sep=";",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LLM-as-judge evaluation.")
    parser.add_argument("--n-selected-cases", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-env", required=True, choices=["aws", "apikey"])
    parser.add_argument("--data-version", required=True)
    parser.add_argument("--data-path", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parse_args()
    raise SystemExit(
        "Run this stage through run_evaluation.py so the client and dataframe "
        "can be passed directly."
    )
