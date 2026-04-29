from utils_prompt import *

approaches = ["Model_Alone", "Model_Studies", "Model_Workflow", "Model_Workflow_Studies"]
approaches_names_map = {"Model_Alone":"LLM-out-of-the-box", "Model_Studies":"RAG-LLM", "Model_Workflow":"WF-LLM", "Model_Workflow_Studies":"WF-RAG-LLM"}

# c1 = list(step_2_criteria.keys())
# c2 = list(step_3_criteria.keys())
# criteria = c1 + c2
# criteria = ["Completeness", "Step-by-step concordance", "Case-tailoring concordance", "Missing data concordance"]

step2_labels = ["Cancer_Surgery", "Imaging_Test", "Therapy"]
step3_labels = ["Biomarker","Biomarker_Result", "Cancer_Surgery", "Grade","Histological_Type","Imaging_Test","Invasion", "Metastasis", "Oncogene", "Response_To_Treatment",
          "Staging","Therapy","Tumor_Finding","Tumor_Size"]