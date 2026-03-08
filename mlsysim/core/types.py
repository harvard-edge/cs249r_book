from typing import Any, Annotated, Union, Optional
from pydantic import AfterValidator, PlainSerializer, BaseModel
from .constants import Q_

def validate_quantity(v: Any) -> Q_:
    if isinstance(v, Q_):
        return v
    if isinstance(v, (int, float, str)):
        try:
            return Q_(v)
        except Exception as e:
            raise ValueError(f"Could not parse Quantity from {v}: {e}")
    raise ValueError(f"Expected Quantity, got {type(v)}")

def serialize_quantity(v: Q_) -> str:
    # Use compact format for serialization
    return f"{v:~P}"

Quantity = Annotated[
    Any,
    AfterValidator(validate_quantity),
    PlainSerializer(serialize_quantity, return_type=str)
]

class Metadata(BaseModel):
    """Provenance information for vetted constants."""
    source_url: Optional[str] = None
    description: Optional[str] = None
    last_verified: Optional[str] = None # YYYY-MM-DD
    version: Optional[str] = None
