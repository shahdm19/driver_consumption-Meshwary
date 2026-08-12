"""Pydantic request/response schemas.

Only the request schema (``TripInput``) was in the original code; the response
shape is built inline in the /predict handler and is kept inline in
``main.py`` to preserve the exact JSON contract.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class TripInput(BaseModel):
    """Request body for ``POST /predict``."""

    make: str
    model: str
    year: int
    road_type: str
    temperature: float
    ac_on: bool
    cc: Optional[float] = None
    from_location: str  # required - used for distance calc + recommendations
    to_location: str    # required - used for distance calc + recommendations
