# METATRON 

## METATRON Project Presentation
METATRON aims to develop an LLM‑based assistant designed to support hepatobiliary (HPB) surgeons during Multidisciplinary Tumor Board discussions by helping them provide treatment recommendations for specific clinical cases. 

**Delivering such recommendations requires a detailed analysis of the clinical cases:** our approach consists of defining and implementing an **expert‑driven structured workflow** that guides the analysis of each clinical case in the identification and extraction of all the relevant information.

## Overview Evaluation Pipeline

This directory contains the evaluation pipeline for the METATRON project. It
provides two complementary analyses:

1. **LLM-based evaluation** compares each generated recommendation with the
   corresponding expert recommendation. An LLM judge scores the responses for
   completeness, step concordance, case tailoring, missing-data concordance,
   and MDT decision concordance, following the provided prompts (**system_prompt_eval** and **user_prompt_eval**).
2. **Named-entity recognition (NER) evaluation** compares entities extracted
   from generated and expert recommendations. It reports semantic metrics
   including TSR, Levenshtein similarity, and exact-match precision.

The evaluation supports the available recommendation approaches:
`Model_Alone` (LLM-baseline), `Model_Studies` (RAG-LLM), `Model_Workflow` (WF-LLM), and
`Model_Workflow_Studies` (WF-RAG-LLM).

## Running the evaluation

Run commands from this directory so that the prompt files and relative output
directories are resolved correctly:

```bash
cd path\to\LLM_based_eval\evaluation
```

### 1. LLM-based evaluation

In our experiment, we used three LLMs as judges: Claude Sonnet 4.6 and Llama 3.3 70B were were accessed via Amazon Bedrock, while GPT-4o was accessed using an API key. 

The entry point is `main_llm_based_evaluation.py`. It:

1. Loads the recommendation comparison CSV for the requested data version --> consider **v8** as the latest version (from the latest version of recommendation generation prompt) 
2. Creates the selected LLM client (AWS Bedrock or an API-key model).
3. Passes the model, client, and data directly to `01_LLM_as_Judge.py`.
4. Writes the judge outputs below a model- and version-specific results
   directory.

For an API-key model:

```bash
uv run main_llm_based_evaluation.py `
  --n-selected-cases 10 `
  --model-env apikey `
  --model-name gpt-4o `
  --data-version v8 `
  --data-path ..\generation\recommendation_strategies\provided_results
```

For AWS Bedrock:

```bash
# Use the model ID as registered on AWS Bedrock. We used: us.anthropic.claude-sonnet-4-6, us.meta.llama3-3-70b-instruct-v1:0
uv run main_llm_based_evaluation.py `
  --n-selected-cases 10 `
  --model-env aws `
  --model-name us.anthropic.claude-sonnet-4-6 
  --data-version v8 `
  --data-path ..\generation\recommendation_strategies\provided_results
```

To use Amazon Bedrock models, the required AWS credentials must be provided in `bedrock_cred/permanent_cred.env`:

AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_MFA_SERIAL=
AWS_DEFAULT_REGION=

AWS workflow additionally runs `00_session_token.py` to obtain the corresponding session token.

For models accessed through the OpenAI API, the API key must be provided in `.env` as:

OPENAI_APIKEY=


### 2. NER evaluation

The entry point is `main_ner_analysis.py`. Before running it, use the provided
`oncological_ner.ipynb` notebook with the John Snow Labs NER pipeline. The
notebook is intended to be opened and executed in **Google Colab**. It extracts
entities from the recommendations and saves the JSON and CSV entity files
required by the analysis script. Resulting CSV and JSON files must be included in `data/<data-version>`

The expected entity layout is:

```text
data\
  v8\
    entities_Model_Alone_Step2.json
    entities_Model_Alone_Step3.json
    entities_df_Model_Alone_Step2.csv
    entities_df_Model_Alone_Step3.csv
    ...
```

After running the notebook and placing the generated files under the
version-specific directory, run:

```bash
uv run main_ner_analysis.py `
  --data-path .\data `
  --data-version v8 `
  --n-selected-cases 10
```

NER results are written to `results_ner\<data-version>\`.

## Results and analysis

All result interpretation and aggregate analyses are collected in
`analyze_results.ipynb`. The analyses can be performed directly using the results already obtained, which are available in the respective result-specific folders.

Generated intermediate prompts are stored in `debug\`. Evaluation outputs are
written to the model-specific result directories and to
`results_ner\<data-version>\` for the NER pipeline.
