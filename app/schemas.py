from pydantic import BaseModel
from typing import List, Optional

class PredictionItem(BaseModel):
    class_name: str
    confidence_score: float

class Details(BaseModel):
    skin_condition: str
    eye_condition: str
    skin_count: int
    eye_count: int

class PredictResponse(BaseModel):
    status: str
    message: Optional[str] = None
    final_conclusion: str
    details: Details
    raw_predictions: List[PredictionItem]