from pydantic import BaseModel
from typing import Literal, List

class ScoreEvaluationOutput(BaseModel):
    Criterion: str
    Score: int
    Explanation: str

class EvaluationOutput(BaseModel):
    #Criterion: str
    LLMEval: Literal["SATISFIED", "NOT SATISFIED"]
    #Explanation: str

class ScoreEvaluation(BaseModel):
    Evaluation: List[ScoreEvaluationOutput]

class SimpleEvaluation(BaseModel):
    Evaluation: List[EvaluationOutput]