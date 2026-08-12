from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Optional

from ..core.units import ureg
from ..core.types import Metadata, Quantity, require_dimensionality, require_unit_family


class DatasetProfile(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)
    name: str
    training_examples: Optional[Quantity] = None
    full_examples: Optional[Quantity] = None
    validation_examples: Optional[Quantity] = None
    test_examples: Optional[Quantity] = None
    num_classes: Optional[int] = None
    languages: Optional[int] = Field(default=None, ge=1)
    keywords: Optional[int] = Field(default=None, ge=1)
    sample_duration: Optional[Quantity] = None
    sample_rate: Optional[Quantity] = None
    sample_width: Optional[Quantity] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    image_channels: Optional[int] = None
    metadata: Metadata = Field(default_factory=Metadata)

    @field_validator("training_examples", "full_examples", "validation_examples", "test_examples", mode="after")
    @classmethod
    def _validate_example_counts(cls, v, info):
        return require_unit_family(v, ureg.count, info.field_name, "count")

    @field_validator("sample_duration", mode="after")
    @classmethod
    def _validate_sample_duration(cls, v, info):
        return require_dimensionality(v, ureg.second, info.field_name)

    @field_validator("sample_rate", mode="after")
    @classmethod
    def _validate_sample_rate(cls, v, info):
        return require_dimensionality(v, 1 / ureg.second, info.field_name)

    @field_validator("sample_width", mode="after")
    @classmethod
    def _validate_sample_width(cls, v, info):
        return require_unit_family(v, ureg.byte, info.field_name, "data")
