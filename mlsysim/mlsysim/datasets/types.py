from pydantic import BaseModel, ConfigDict, Field
from typing import Optional

from ..core.types import Metadata, Quantity


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
