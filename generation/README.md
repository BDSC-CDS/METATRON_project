# METATRON 

## METATRON Project Presentation
METATRON aims to develop an LLM‑based assistant designed to support hepatobiliary (HPB) surgeons during Multidisciplinary Tumor Board discussions by helping them provide treatment recommendations for specific clinical cases. 

**Delivering such recommendations requires a detailed analysis of the clinical cases:** our approach consists of defining and implementing an **expert‑driven structured workflow** that guides the analysis of each clinical case in the identification and extraction of all the relevant information.

## Overview Generation Pipeline

The `generation/` directory contains the two-stage pipeline used to generate
LLM-based treatment recommendations for colorectal liver metastasis (CRLM) cases:

1. **`workflow_based_case_analysis`** performs structured information
   extraction from each clinical case using the expert-defined workflow.
2. **`recommendation_strategies`** uses the case text, the extracted workflow
   information, and retrieved literature references to generate and validate
   treatment recommendations.

## Pipeline steps and outputs

### 1. Workflow-based case analysis

[`workflow_based_case_analysis/main_workflow_executor.py`](workflow_based_case_analysis/main_workflow_executor.py)
loads `workflow/workflow_information_extraction.json` and evaluates every case through `Step1` to `Step4`. Conditions determine which workflow items are
evaluated; leaf items trigger structured information extraction prompts.

The step writes:

- one text file per case and workflow step in the directory supplied through `--structured-information-path`, named
  `structured_information_<case-number>_<step>.txt`;
- `conditional_steps_results.csv` and one `sie_<item>.csv` per extracted
  feature in the directory supplied through `--output-dir`;
- `debug_file.txt` in the results directory.

These structured-information files are consumed by the recommendation stage.
The input CSV must be semicolon-separated and contain `Case_Number` and
`Content` columns.

### 2. Recommendation strategies

[`recommendation_strategies/main_generate_recommendation.py`](recommendation_strategies/main_generate_recommendation.py) runs four prompting strategies:

- model alone (LLM-baseline);
- model with RAG literature excerpts (RAG-LLM);
- model with workflow-based structured information (WF-LLM);
- model with both workflow information and RAG excerpts (WF-RAG-LLM).

The script then checks whether each generated recommendation and expert recommendation contains `Step 1`, `Step 2`, and `Step 3`.

It writes the following files to `--output-dir`:

- `recommendation_results_version_<prompt-version>.csv`: generated model
  recommendations;
- `recommendation_steps_<prompt-version>.csv`: extracted recommendation steps;
- `recommendation_comparison_<prompt-version>.csv`: model recommendations
  combined with expert recommendations;
- `step_errors_<prompt-version>.csv`: missing or malformed step information.

The last prompt version is `v8`; the full dataset contains 110 cases. A smaller
dataset is available in `../data/data_short.csv`; use `test_results` as the output
directory for test runs.

## Model configuration

Both stages uses the OpenAI Python client to communicate with the configured LLM and embedding model endpoints. By default, both models are served by a local Ollama instance through its OpenAI-compatible endpoints.

You can configure the model URL and model name in the relevant `models.py` file in each of the two directories according to your local setup.

The models.py file contains the configuration for both the LLM and the embedding model:

```
models = {
    "llm": {
        "url": "http://localhost:11434/v1",
        "model_name": "gpt-oss:120b-cloud"
    },
    "embedder": {
        "url": "http://localhost:11434/v1",
        "model_name": "qwen3-embedding:0.6b"
    }
}
```

By default, both models are served through a local Ollama instance using its OpenAI-compatible API (port 11434).

```
LLM: gpt-oss:120b-cloud
Embedding model: qwen3-embedding:0.6b
API endpoint: http://localhost:11434/v1
```

If you use a different model or an alternative inference server (e.g., vLLM), update the corresponding url and model_name values in models.py accordingly.

## Running the complete pipeline

Run these commands from the `generation/` directory. First, generate the
workflow-based structured information:

```bash
# Comments about the input args
# Use test_structured_information_files to save your structured analyses. 
# The original files containing the structured analyses of all 110 cases are provided in structured_information_files
uv run workflow_based_case_analysis/main_workflow_executor.py \
  --cases data/HPB_cases.csv \
  --structured-information-path test_structured_information_files \
  --output-dir workflow_based_case_analysis/results
```

Then generate and validate the recommendations:

```bash
# Comments about the input args
# v8 is the latest prompt version.
# As structured analyses dir, you can choose your files in test_structured_information_files or the original files in structured_information_files
uv run recommendation_strategies/main_generate_recommendation.py \
  --cases data/HPB_cases.csv \
  --structured-information-dir test_structured_information_files \
  --expert-recommendation data/expert_recommendations_v8.csv \
  --prompt-version v8 \
  --output-dir recommendation_strategies/test_results
```

For a smaller test run, use `data/data_short.csv`.

See the
[`workflow_based_case_analysis/README.md`](workflow_based_case_analysis/README.md)
and
[`recommendation_strategies/README.md`](recommendation_strategies/README.md)
for stage-specific details.