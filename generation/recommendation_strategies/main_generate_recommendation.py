import argparse
import os
from pathlib import Path

import pandas as pd
from openai import OpenAI

from models import models
from utils import clean_answer, combine_files, find_files
from utils_call_models import query_model_structured_message

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT_DIR = SCRIPT_DIR / "prompt"
DEFAULT_RAG_DIR = SCRIPT_DIR / "../rag_results"
DEFAULT_DEBUG_DIR = SCRIPT_DIR / "debug"

RECOMMENDATION_COLUMNS = [
    "Case_Number",
    "Content",
    "Model_Alone",
    "Model_Studies",
    "Model_Workflow",
    "Model_Workflow_Studies",
]
APPROACHES = (
    ("Model_Alone", "LLM out-of-the-box"),
    ("Model_Studies", "LLM + RAG"),
    ("Model_Workflow", "LLM + Workflow"),
    ("Model_Workflow_Studies", "LLM + Workflow + RAG"),
)


def generate_recommendations(
    cases_path: str | os.PathLike[str],
    structured_information_dir: str | os.PathLike[str],
    prompt_version: str,
    output_dir: str | os.PathLike[str],
) -> Path:
    """Generate recommendations for every case and save them to ``output_path``."""
    prompt_dir = DEFAULT_PROMPT_DIR
    rag_dir = DEFAULT_RAG_DIR
    debug_dir = DEFAULT_DEBUG_DIR
    debug_dir.mkdir(parents=True, exist_ok=True)

    cases = pd.read_csv(cases_path, encoding="utf-8", sep=";")
    print(f"No. cases: {len(cases)}, No. categories: {len(set(cases['Category']))}")

    def read_prompt(filename: str) -> str:
        with open(prompt_dir / filename, "r", encoding="utf-8") as prompt_file:
            return prompt_file.read()

    system_prompt_a_12 = read_prompt("system_prompt_Approach_1_2.txt")
    system_prompt_a_34 = read_prompt("system_prompt_Approach_3_4.txt")
    user_prompts = {
        "Model_Alone": read_prompt(
            f"recommendation_prompt_Approach_1_{prompt_version}.txt"
        ),
        "Model_Studies": read_prompt(
            f"recommendation_prompt_Approach_2_{prompt_version}.txt"
        ),
        "Model_Workflow": read_prompt(
            f"recommendation_prompt_Approach_3_{prompt_version}.txt"
        ),
        "Model_Workflow_Studies": read_prompt(
            f"recommendation_prompt_Approach_4_{prompt_version}.txt"
        ),
    }
    client = OpenAI(api_key="empty", base_url=models["llm"]["url"])
    recommendation_results = pd.DataFrame(columns=RECOMMENDATION_COLUMNS)

    for _, case in cases.iterrows():
        case_number = case["Case_Number"]
        case_content = case["Content"]
        print(f"Case: {case_number}")
        row = {column: None for column in RECOMMENDATION_COLUMNS}
        row["Case_Number"] = case_number
        row["Content"] = case_content

        structured_information_path = str(structured_information_dir) + os.sep
        structured_files = find_files(case_number, structured_information_path)
        has_workflow = bool(structured_files)
        structured_content = (
            combine_files(structured_files, structured_information_path)
            if has_workflow
            else ""
        )
        # Option 1: Read references from file
        refs_path = rag_dir / f"paragraphs_Approach2_{case_number}.txt"
        with open(refs_path, "r", encoding="utf-8") as refs_file:
            refs = refs_file.read()

        # Option 2: Generate references using RAG (if needed)
        # from utils_rag import build_excerpts
        # rag_prompt = rag_prompt_2.format(clinical_case=case_content)
        # refs, sims = build_excerpts(rag_prompt, "", 30)  # empty filter
        # refs = json.dumps(refs, indent=4)
        # f3.write(refs)
        # f3.close()

        debug_prompt_path = debug_dir / f"debug_prompt_case{case_number}.txt"
        debug_answer_path = debug_dir / f"debug_answer_case{case_number}.txt"
        with (
            open(debug_prompt_path, "w", encoding="utf-8") as debug_prompt,
            open(debug_answer_path, "w", encoding="utf-8") as debug_answer,
        ):
            for approach, approach_name in APPROACHES:
                if approach in ("Model_Workflow", "Model_Workflow_Studies") and not has_workflow:
                    print(f"No structured information found for case {case_number}!")
                    continue

                if approach == "Model_Alone":
                    prompt = user_prompts[approach].format(
                        clinical_case=case_content
                    )
                    system_prompt = system_prompt_a_12
                elif approach == "Model_Studies":
                    prompt = user_prompts[approach].format(
                        clinical_case=case_content, refs=refs
                    )
                    system_prompt = system_prompt_a_12
                elif approach == "Model_Workflow":
                    prompt = user_prompts[approach].format(
                        clinical_case=case_content,
                        str_analysis=structured_content,
                    )
                    system_prompt = system_prompt_a_34
                else:
                    prompt = user_prompts[approach].format(
                        clinical_case=case_content,
                        str_analysis=structured_content,
                        refs=refs,
                    )
                    system_prompt = system_prompt_a_34

                debug_prompt.write(
                    f"{approach_name}\n{system_prompt}\n{prompt}\n"
                    "---------------------------------------------------\n"
                )
                answer = query_model_structured_message(
                    client, system_prompt, prompt
                )
                debug_answer.write(
                    f"{approach_name}\n{answer}\n"
                    "---------------------------------------------------\n"
                )
                row[approach] = clean_answer(answer)

        recommendation_results.loc[len(recommendation_results)] = row

    output_path = Path(output_dir) / f"recommendation_results_version_{prompt_version}.csv"
    recommendation_results.to_csv(output_path, index=False, sep=";")
    return output_path


def check_step_structure(
    recommendation_results_path: str | os.PathLike[str],
    expert_recommendation_path: str | os.PathLike[str],
    prompt_version: str,
    output_dir: str | os.PathLike[str],

) -> dict[str, Path]:
    """Combine model and expert recommendations and extract their three steps."""
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    data_model = pd.read_csv(
        recommendation_results_path, sep=";", encoding="utf-8"
    )
    data_expert = pd.read_csv(
        expert_recommendation_path, sep=";", encoding="utf-8"
    )
    expert_by_case = data_expert.set_index("Case_Number")["Expert_Recommendation"]
    missing_expert = sorted(
        set(data_model["Case_Number"]) - set(expert_by_case.index)
    )
    if missing_expert:
        raise ValueError(f"Missing expert recommendations for cases: {missing_expert}")

    data_final = data_model.copy()
    data_final["Expert"] = data_final["Case_Number"].map(expert_by_case)
    recommendation_columns = list(data_final.columns[2:])
    keywords = ("Step 1", "Step 2", "Step 3")
    step_columns = [
        f"{column}_{step}"
        for column in recommendation_columns
        for step in keywords
    ]
    data_steps = pd.DataFrame(
        index=data_final["Case_Number"], columns=step_columns
    )
    errors = {column: [] for column in recommendation_columns}

    for case_number, row in data_final.iterrows():
        print(f"Case: {row['Case_Number']}")
        for column in recommendation_columns:
            value = row[column]
            if not isinstance(value, str):
                errors[column].append(row["Case_Number"])
                continue
            cleaned = (
                value.replace("\u202f", " ")
                .replace("\u00a0", " ")
                .replace("\u2009", " ")
            )
            positions = [cleaned.find(keyword) for keyword in keywords]
            if -1 in positions:
                errors[column].append(row["Case_Number"])
                print(
                    f"Issues in steps identification for column {column}, "
                    f"case {row['Case_Number']}"
                )
                continue
            data_steps.loc[row["Case_Number"], f"{column}_Step 1"] = cleaned[
                positions[0] : positions[1]
            ]
            data_steps.loc[row["Case_Number"], f"{column}_Step 2"] = cleaned[
                positions[1] : positions[2]
            ]
            data_steps.loc[row["Case_Number"], f"{column}_Step 3"] = cleaned[
                positions[2] :
            ]
            data_final.at[case_number, column] = cleaned

    step_error_table = pd.DataFrame(
        {
            "N_Not_Correct_Structure": {
                column: len(case_numbers)
                for column, case_numbers in errors.items()
            },
            "Case_Not_Correct_Structure": errors,
        }
    )
    paths = {
        "steps": output_directory
        / f"recommendation_steps_{prompt_version}.csv",
        "comparison": output_directory
        / f"recommendation_comparison_{prompt_version}.csv",
        "errors": output_directory
        / f"step_errors_{prompt_version}.csv",
    }
    data_steps.to_csv(paths["steps"], encoding="utf-8", sep=";")
    data_final.to_csv(paths["comparison"], encoding="utf-8", index=False, sep=";")
    step_error_table.to_csv(paths["errors"], encoding="utf-8", sep=";")
    return paths


def main(
    cases: str,
    structured_information_dir: str,
    expert_recommendation: str,
    prompt_version: str,
    output_dir: str,
) -> None:
    """Run recommendation generation followed by step-structure validation."""
    print("1. Generate recommendation")
    recommendation_path = generate_recommendations(
        cases,
        structured_information_dir,
        prompt_version,
        output_dir,
    )
    print("2. Check step-based structure")
    check_step_structure(
        recommendation_path,
        expert_recommendation,
        prompt_version,
        output_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate recommendations and check their step-based structure."
    )
    parser.add_argument("--cases", required=True, help="Complete path to the cases CSV.")
    parser.add_argument(
        "--structured-information-dir",
        required=True,
        help="Directory containing the structured-information files.",
    )
    parser.add_argument(
        "--expert-recommendation",
        required=True,
        help="Complete path to the expert recommendations CSV.",
    )
    parser.add_argument("--prompt-version", required=True, help="Prompt version, e.g. v8.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Path for output files.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(**vars(parse_args()))
