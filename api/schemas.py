from typing import List, Optional

from pydantic import BaseModel, Field


class DetectionBox(BaseModel):
    x_min: float = Field(..., ge=0.0, le=1.0)
    y_min: float = Field(..., ge=0.0, le=1.0)
    x_max: float = Field(..., ge=0.0, le=1.0)
    y_max: float = Field(..., ge=0.0, le=1.0)
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class Recommendation(BaseModel):
    action: str
    details: str
    priority: str = "medium"


class PredictionResult(BaseModel):
    crop: str
    problem_type: str
    detected_problem: str
    scientific_name: Optional[str] = None
    severity: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    recommendations: List[Recommendation]
    boxes: List[DetectionBox] = Field(default_factory=list)
