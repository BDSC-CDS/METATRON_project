import argparse
import json
import random
from pathlib import Path

import pandas as pd

from utils_metrics import choose_fun


APPROACHES = ["Model_Alone", "Model_Studies", "Model_Workflow", "Model_Workflow_Studies"]
STEPS = ["Step2", "Step3"]
METRICS = ["TSR", "Levenshtein", "Exact_Match_P"]


def main(data_path: str, data_version: str, n_selected_cases: int) -> None:
    """Calculate NER metrics for a reproducible selection of cases."""
    if n_selected_cases < 1:
        raise ValueError("n_selected_cases must be greater than zero.")

    entities_dir = Path(data_path) / data_version
    result_dir = Path("results_ner") / data_version
    result_dir.mkdir(parents=True, exist_ok=True)

    first_data = pd.read_csv(
        entities_dir / f"entities_df_{APPROACHES[0]}_{STEPS[0]}.csv"
    )
    available_cases = sorted(first_data["Case_N"].unique().tolist())
    if n_selected_cases > len(available_cases):
        raise ValueError(
            f"n_selected_cases ({n_selected_cases}) cannot exceed "
            f"the number of available cases ({len(available_cases)})."
        )

    random.seed(42)
    selected_cases = set(random.sample(available_cases, n_selected_cases))

    for step in STEPS:
        for approach in APPROACHES:
            with open(
                entities_dir / f"entities_{approach}_{step}.json",
                "r",
                encoding="utf-8",
            ) as entity_file:
                entity_dict = json.load(entity_file)
            ner_data = pd.read_csv(
                entities_dir / f"entities_df_{approach}_{step}.csv"
            )
            ner_data = ner_data[ner_data["Case_N"].isin(selected_cases)]

            content_res = pd.DataFrame(
                columns=["Case_N", "Label", "Metric", "Value"]
            )
            labels = sorted(ner_data["Entity_Label"].unique().tolist())

            for case in sorted(selected_cases):
                for label in labels:
                    case_entities = entity_dict[str(case)]["Text"][label]
                    reference_entities = entity_dict[str(case)]["Ref_Text"][label]
                    for metric in METRICS:
                        value = choose_fun(
                            metric, reference_entities, case_entities
                        )
                        content_res.loc[len(content_res)] = [
                            case,
                            label,
                            metric,
                            value,
                        ]

            content_res.to_csv(
                result_dir / f"NER_semantic_metric_{approach}_{step}.csv",
                index=False,
                sep=";",
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate NER semantic metrics.")
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path containing the version-specific entity data directories.",
    )
    parser.add_argument("--data-version", required=True)
    parser.add_argument("--n-selected-cases", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main(**vars(parse_args()))
