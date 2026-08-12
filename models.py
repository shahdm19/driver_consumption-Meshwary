from __future__ import annotations
from typing import Optional
from pydantic import BaseModel

class TripInput(BaseModel):

    make: str
    model: str
    year: int
    road_type: str
    temperature: float
    ac_on: bool
    cc: Optional[float] = None
    from_location: str  
    to_location: str    
