from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Optional

from ..core.constants import ureg
from ..core.types import Metadata, Quantity, require_unit_family


class DatasetProfile(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    name: str
    training_examples: Optional[Quantity] = None
    test_examples: Optional[Quantity] = None
    num_classes: Optional[int] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    image_channels: Optional[int] = None
    metadata: Metadata = Field(default_factory=Metadata)

    @field_validator("training_examples", "test_examples", mode="after")
    @classmethod
    def _validate_example_counts(cls, v, info):
        return require_unit_family(v, ureg.count, info.field_name, "count")
