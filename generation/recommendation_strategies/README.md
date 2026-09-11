# METATRON 

## METATRON Project Presentation
METATRON aims to develop an LLM‑based assistant designed to support hepatobiliary (HPB) surgeons during Multidisciplinary Tumor Board discussions by helping them provide treatment recommendations for specific clinical cases. 

### Repo: generation/recommendation_strategies 

#### Overview
This repository contains code for generating treatment recommendations for **Colorectal Liver Metastasis (CRLM)** cases using Large Language Models (LLMs).  The core script is **`main_generate_recommendation.py`**, which orchestrates four different prompting strategies:

| Approach | Description |
|----------|-------------|
| **1️⃣ Model‑Alone (LLM-baseline)** | Direct LLM generation using only the raw clinical case. |
| **2️⃣ Model + RAG (RAG-LLM)** | The LLM is supplied with retrieved study excerpts (RAG – Retrieval‑Augmented Generation). |
| **3️⃣ Model + Workflow (WF-LLM)** | The LLM receives a structured analysis of the case (from a previous step, implemented in `../workflow_based_case_analysis` dir). |
| **4️⃣ Model + Workflow + RAG (WF-RAG-LLM)** | Combines the structured workflow with filtered study excerpts. |

All the prompt used for the four approaches are provided in `prompt`. 

---
#### Recommendation pipeline

The recommendation script runs two steps in sequence:

1. **Generate recommendations:** runs the four prompting strategies for every case in
   `--cases` and saves the model outputs to
   `recommendation_results_version_<prompt-version>.csv`.
2. **Check step-based structure:** reads that recommendation file and the
   `--expert-recommendation` file, extracts `Step 1`, `Step 2`, and `Step 3` from
   each recommendation, and records missing or malformed steps. It produces:
   - `recommendation_steps_<prompt-version>.csv` (extracted steps);
   - `recommendation_comparison_<prompt-version>.csv` (model and expert recommendations);
   - `step_errors_<prompt-version>.csv` (structure errors by column and case).

All files are written to the directory passed through `--output-dir`.

#### Running the pipeline

From the `generation/recommendation_strategies` directory, run:

```bash
# Some comments about the input args
# v8 is the latest prompt version.
# For a smaller dataset, replace ../data/HBP_cases.csv with ../data/data_short.csv/.
# results dir contains recommendations for all 110 cases, as reference.
# Use test_results as --output-dir when running a test.
# As structured_information_files dir, you can choose to use your test_structured_information_files or the original structured_information_files
uv run main_generate_recommendation.py \
  --cases ../data/HPB_cases.csv \
  --structured-information-dir ../test_structured_information_files \
  --expert-recommendation ../data/expert_recommendations_v8.csv \
  --prompt-version v8 \
  --output-dir test_results
```

Before running the workflow, update `models.py` with the instance of the local model you want to use, such as a vLLM or Ollama server, including its URL and model name.

---

## Additional files 
- **`rag_statistics.py`** computes quantitative statistics about which literature papers and topics are retrieved by the RAG component for each of the two RAG‑enabled prompting approaches. 

---
