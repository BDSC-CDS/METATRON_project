from pydantic import BaseModel
from typing import Literal, List
from enum import Enum

# class EvaluationOutput(BaseModel):
#     #Criterion: str
#     LLMEval: Literal["SATISFIED", "NOT SATISFIED"]
#     #Explanation: str

class EvaluationOutput(BaseModel):
    completeness: Literal[0,1]
    step_concordance: Literal[0,1]
    case_tailoring: Literal[0,1]   
    missing_data_concordance: Literal[0,1]

class EvaluationOutputReasoning(BaseModel):
    completeness: Literal[0,1]
    step_concordance: Literal[0,1]
    case_tailoring: Literal[0,1]   
    missing_data_concordance: Literal[0,1]
    reasoning: str