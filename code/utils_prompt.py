# Prompt with roles (junior and senior): this version uses a step-wise approach (one prompt for each criterion)
# system_prompt = '''
# You are an expert hepatobiliary surgeon, expected to assess the factual alignment between two recommendations regarding next-step management in patients diagnosed with a primary colorectal tumor and liver metastases, including the potential presence of additional extrahepatic metastases. 
# One recommendation is provided by a senior expert colleague (reference recommendation), and the other by a junior colleague whose work should be evaluated.
# '''

# completeness = '''Does the junior recommendation address all the major clinical domains and management components covered in the reference recommendation, without significant omissions?
# Evaluate this criterion considering these categories, including but not limited to:
# - Resectability assessment and its rationale
# - Preoperative staging (e.g. imaging exams)
# - Systemic therapy 
# - Surgical management 
# - Postoperative therapy and surveillance.
# '''

# step_concordance = '''Does the junior recommendation reproduce the same key clinical reasoning steps and management decisions as the reference recommendation?
# Evaluate this criterion considering the two steps of the recommendations:
# - Step 1: Do the junior and reference recommendations provide the same assessment of resectability? (e.g. upfront resectable, potentially resectable, not resectable)
# - Step 2: Do the junior and reference recommendations propose the same management plan? Consider the sequence of proposed actions and their content, across categories such as imaging/staging, systemic therapy, and surgery.
# '''

# case_tailoring = '''Does the junior recommendation include the same case-specific clinical details and patient-related factors identified in the reference recommendation?
# Evaluate this criterion considering these examples of case-specific elements, including — but not limited to:
# - Molecular profile implications 
# - Lesion-specific surgical considerations
# - Case-specific surveillance targets (e.g. CEA threshold, restaging timing)
# - Patient-level factors relevant to treatment sequencing (e.g. nutritional status)
# '''

# step_2_criteria = {"Completeness":completeness,"Step-by-step concordance":step_concordance,"Case-tailoring concordance":case_tailoring}
# step_3_criteria = {"Missing data concordance":"Does the junior recommendation identify the same essential missing clinical information highlighted in the reference recommendation?"}

# task = '''You are given the junior recommendation and the reference recommendation. Your task is to indicate whether the specified evaluation criterion is satisfied by the junior recommendation.

# Criterion: {criterion}

# Reference recommendation: {reference}
# '''

############################
# Prompt without roles (junior and senior): this version uses one single prompt to evaluate all criteria at once
system_prompt = '''
You are an expert hepatobiliary surgeon, expected to assess the factual alignment between two recommendations regarding next-step management in patients diagnosed with a primary colorectal tumor and liver metastases, including the potential presence of additional extrahepatic metastases. 
'''

completeness = '''Compare the two recommendations and determine whether they cover the same major clinical domains and management components without significant differences.
Evaluate this criterion considering these categories, including but not limited to:
- Resectability assessment and its rationale
- Preoperative staging (e.g. imaging exams)
- Systemic therapy 
- Surgical management 
- Postoperative therapy and surveillance.
'''

step_concordance = '''Compare the two recommendations and determine whether they reproduce the same key clinical reasoning steps and management decisions.
Evaluate this criterion considering the two steps of the recommendations:
- Step 1: Do the two recommendations provide the same assessment of resectability? (e.g. upfront resectable, potentially resectable, not resectable)
- Step 2: Do the two recommendations propose the same management plan? Consider the sequence of proposed actions and their content, across categories such as imaging/staging, systemic therapy, and surgery.
'''

case_tailoring = '''Do the two recommendations include the same case-specific clinical details and patient-related factors?
Evaluate this criterion considering these examples of case-specific elements, including — but not limited to:
- Molecular profile implications 
- Lesion-specific surgical considerations
- Case-specific surveillance targets (e.g. CEA threshold, restaging timing)
- Patient-level factors relevant to treatment sequencing (e.g. nutritional status)
'''

missing_data_concordance = '''Do the two recommendations identify the same essential missing clinical information? For this evaluation, focus on Step 3 of the recommendations.'''

reasoning = '''- Support your evaluation with a brief reasoning that justifies your scores for each criterion, based on the content of the two recommendations.'''

task = '''You are given the two recommendations. Your task is to indicate whether they can be considered substantially similar considering the following criteria: completeness, step-by-step concordance, case-tailoring concordance, and missing data concordance.

Criteria description: 
Completeness: {completeness}

Step-by-step concordance: {step_concordance}

Case-tailoring concordance: {case_tailoring}

Missing data concordance: {missing_data_concordance}

For each criterion, use the score 1 to indicate that the two recommendations are aligned, i.e. they satisfy the criterion; otherwise, use the score 0 to indicate that they are not aligned. 
Instructions:
- Rely on the criteria description for your evaluation. 
- Follow the output format required. 

Recommendation 1: 
{r1}

Recommendation 2: 
{r2}

Output format:
{{
    "completeness": 0 or 1,
    "step_concordance": 0 or 1, 
    "case_tailoring": 0 or 1,
    "missing_data_concordance": 0 or 1
}}
'''