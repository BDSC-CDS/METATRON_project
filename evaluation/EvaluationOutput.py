from pydantic import BaseModel
from typing import Literal, List
from enum import Enum

class Decision(BaseModel):
    oncological_strategy: str
    systemic_therapy: str
    surgical_strategy: str
    timing_of_surgery: str
    staging_restaging: str
    anatomical_compliance: str
    critical_missing_data: str
    final_mdt_decision: str

class EvaluationOutput(BaseModel):
    completeness: Literal[1,2,3,4,5]
    step_concordance: Literal[1,2,3,4,5]
    case_tailoring: Literal[1,2,3,4,5]   
    missing_data_concordance: Literal[1,2,3,4,5]
    mdt_decision_concordance: Literal[1,2,3,4,5]

class EvaluationOutputReasoning(BaseModel):
    completeness: Literal[1,2,3,4,5]
    step_concordance: Literal[1,2,3,4,5]
    case_tailoring: Literal[1,2,3,4,5]   
    missing_data_concordance: Literal[1,2,3,4,5]
    mdt_decision_concordance: Literal[1,2,3,4,5]
    reasoning: str

class OutputEval(BaseModel):
    expert_decisions: Decision
    candidate_decisions: Decision
    evaluation: EvaluationOutputReasoning
