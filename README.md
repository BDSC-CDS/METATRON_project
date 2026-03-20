# Automatic Evaluation of LLM-Driven Oncological Recommendations - METATRON Project

This repository performs automatic evaluation of LLM-generated answers against reference (expert) answers using an LLM-as-judge framework.

The evaluation is carried out in two ways:
1. Whole answer judgment, where the full generated answer is scored against the expert reference.
2. Step-wise judgment, where the generated answer is split into substeps and each is evaluated individually, then aggregated.

It also performs **semantic analysis** by comparing structured information extracted from texts.
Extraction sources include:
- LLM-based structured extraction using a domain-specific taxonomy (e.g., resectability assessment labels).
- Zero-shot clinical NER models from the John Snow Labs library (oncology-focused entity extraction).

The LLM-as-judge workflow uses high-performance models via API calls and AWS Bedrock.


## Code description

- `00_arrange_data.py` - dataset preparation 
- `00_session_token.py` - AWS bedrock session token handling
- `01_LLM_as_Judge_OneText.py` - one-text LLM judging 
- `01_LLM_as_Judge_OneText_bedrock.py` - Bedrock-specific variant for one-text evaluation
- `01_LLM_as_judge_StepWise.py` - multi-step evaluation/prompts to improve concordance
- `01B_LLM_as_judge_table_results.py` - result formatting and table reporting
- `02_ner_analysis.py` - NER-centric analysis (based on pre-collected entities)
- `EvaluationOutput.py` - Pydantic class for structured LLM-based evaluation
- `utils_call_bedrock_models.py` - Bedrock API call wrappers
- `utils_metrics.py` - metric calculation utilities (accuracy, concordance, etc.)
- `utils_prompt.py` - prompt builder helpers and prompt templates
- `utils_variables.py` - common constants and configuration values

- `analyze_results.ipynb` - exploratory data analysis notebook for LLM evaluation results, including metrics visualization, concordance summaries, and model comparison plots.


