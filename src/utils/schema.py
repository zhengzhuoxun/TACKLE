from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ProjectModel(BaseModel):
    """Shared Pydantic base model for validated project data structures."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    def to_dict(self) -> dict:
        return self.model_dump(mode="python")

