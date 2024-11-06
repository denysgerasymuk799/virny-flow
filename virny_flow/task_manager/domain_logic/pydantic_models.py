from pydantic import BaseModel
from typing import List, Optional


class ObservationModel(BaseModel):
    config: dict
    objectives: List[float]
    constraints: Optional[List[float]] = None
    trial_state: Optional[str] = "SUCCESS"
    elapsed_time: Optional[float] = None
    extra_info: Optional[dict] = None
