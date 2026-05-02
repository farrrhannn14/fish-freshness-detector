from pydantic import BaseModel
from typing import List, Optional

class DetectionDetail(BaseModel):
    id_fitur: int
    yolo_class: str
    yolo_confidence: float
    efficientnet_prediction: str
    efficientnet_confidence: float
    koordinat: dict

class PredictionResponse(BaseModel):
    filename: str
    warning: Optional[str] = None
    status_kesimpulan: str
    total_fitur_terdeteksi: int
    detail_deteksi: List[DetectionDetail]