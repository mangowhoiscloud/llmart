"""Pipeline state schema."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PipelineState(BaseModel):
    """Immutable state flowing through the LLMART pipeline."""

    candidates: list[dict[str, Any]] = Field(default_factory=list)
    stage: str = "init"
    top_k: int = 30
    errors: list[str] = Field(default_factory=list)
