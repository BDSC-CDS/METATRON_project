# METATRON

## METATRON Project Presentation
METATRON aims to develop an LLM‑based assistant designed to support hepatobiliary (HPB) surgeons during Multidisciplinary Tumor Board discussions by helping them provide treatment recommendations for specific clinical cases. 

**Delivering such recommendations requires a detailed analysis of the clinical cases:** our approach consists of defining and implementing an **expert‑driven structured workflow** that guides the analysis of each clinical case in the identification and extraction of all the relevant information.

### Repo: generation/workflow_based_case_analysis 

#### Workflow Description
This workflow adopts an analytical, hierarchical structure of questions, and information elements to be evaluated, formalized as a JSON schema with a tree‑like organization.
It is composed of three types of nodes:

- **Step nodes**, representing the major domains of the evaluation (e.g., Primary Tumor, Liver Metastases, Extrahepatic Metastases, Anatomical feasibility for surgery).
- **Item nodes**, listing the specific pieces of information to be extracted from the clinical case. Item nodes can contain sub‑Items, each belonging to a shared evaluation category. The sub‑Items represent the leaf nodes of the hierarchy, encapsulating the final, granular data elements to be analyzed.
- **Condition nodes**, defining prerequisite questions that must be verified before a group of Items is assessed, ensuring that the reasoning path follows clinically appropriate decision points.

The resulting structure guides the clinical case analysis in a systematic, stepwise manner, mirroring the reasoning process of HPB specialists and ensuring a transparent and reproducible decision‑making workflow.


#### Code Description

`main_workflow_executor.py` is the entry point for running the structured information extraction workflow on clinical case reports. It accepts three required arguments: the complete cases CSV path, the complete structured-information output path, and the results output directory. It:

- Loads a workflow definition from `workflow/workflow_information_extraction.json` which specifies the steps and sub‑steps for information extraction.
- Reads the clinical cases from the complete CSV path provided with `--cases`.
- Iterates over each clinical case and, for each defined step (Step1, Step2, Step3, Step4), invokes the recursive `execute_step` function from `utils_workflow_steps.py`. `execute_step` walks the workflow tree, evaluates any **Condition** nodes, and:
  - If a condition is true, it continues deeper into the sub‑items.
  - If no condition is present, it performs a Structured Information Extraction (SIE) step, sending a tailored LLM prompt, parsing the response, and storing the extracted features in a results table.
  - It also logs prompt usage (simple vs. structured).
- Saves textual structured information to the directory provided with `--structured-information-path` and tabular results to the directory provided with `--output-dir`. These files are used in the second phase of recommendation generation (`recommendation_strategies`).

Before running the workflow, update `models.py` with the instance of the local model you want to use, such as a vLLM or Ollama server, including its URL and model name.

#### Running the Workflow with uv

From this directory, run the workflow with the bundled dataset. All three paths are supplied explicitly:

```bash
# Some comments about the input args
# For a smaller dataset, replace ../data/HBP_cases.csv with ../data/data_short.csv/.
# Use test_structured_information_files to save structured analyses, since the original dir structured_information_files contains the structured for all 110 cases to be used as reference.

uv run main_workflow_executor.py \
  --cases ../data/HPB_cases.csv \
  --structured-information-path ../test_structured_information_files \
  --output-dir results
```

Note: if you want to use your own data, consider that the input CSV must use `;` as its separator and contain the required `Case_Number` and `Content` columns. The workflow definition is loaded from `workflow/workflow_information_extraction.json`; run `uv run main_workflow_executor.py --help` for details.
