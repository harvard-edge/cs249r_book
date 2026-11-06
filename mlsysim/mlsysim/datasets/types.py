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
    episodes: Optional[Quantity] = None
    total_duration: Optional[Quantity] = None
    embodiments: Optional[int] = None
    tasks: Optional[int] = None
    # Tabular click-log feature counts (added 2026-10-20): integer-valued
    # (dense) and categorical columns per example, e.g. Criteo's 13 and 26.
    dense_features: Optional[int] = Field(default=None, ge=1)
    categorical_features: Optional[int] = Field(default=None, ge=1)
    # Size of the distributed (compressed) training-split archive, e.g. the
    # ILSVRC-2012 training tarball (added 2026-09-28).
    training_archive_size: Optional[Quantity] = None
    metadata: Metadata = Field(default_factory=Metadata)

    @field_validator("training_archive_size", mode="after")
    @classmethod
    def _validate_training_archive_size(cls, v):
        if v is None:
            return v
        return require_unit_family(v, ureg.byte, "training_archive_size", "data")

    @property
    def mean_compressed_image_size(self) -> Optional[Quantity]:
        """Average the distributed training archive over its examples.

        Returns
        -------
        Quantity or None
            Mean compressed bytes per training example; None if archive size or example count is absent.

        Notes
        -----
        Equation: mean_bytes = training_archive_size / training_examples.

        Assumptions: the archive and count describe the same training split, with a positive count. Archive container overhead is included in the average.

        Excludes: decoded image tensors, variation among examples, augmentation buffers, and validation/test splits.

        References
        ----------
        Derivation: divide total stored bytes by the number of archived examples. Despite the legacy image name, the arithmetic does not inspect image content.

        Examples
        --------
        >>> from mlsysim.datasets.types import DatasetProfile
        >>> from mlsysim.core.units import Q_
        >>> data = DatasetProfile(name='Archive', training_archive_size=Q_('10 MB'), training_examples=Q_(100, 'count'))
        >>> data.mean_compressed_image_size * data.training_examples.magnitude == data.training_archive_size
        True
        """
        if self.training_archive_size is None or self.training_examples is None:
            return None
        n = self.training_examples.to(ureg.count).magnitude
        return (self.training_archive_size / n).to(ureg.byte)

    @field_validator("training_examples", "full_examples", "validation_examples", "test_examples", "episodes", mode="after")
    @classmethod
    def _validate_example_counts(cls, v, info):
        if v is None:
            return v
        return require_unit_family(v, ureg.count, info.field_name, "count")

    @field_validator("sample_duration", "total_duration", mode="after")
    @classmethod
    def _validate_sample_duration(cls, v, info):
        if v is None:
            return v
        return require_dimensionality(v, ureg.second, info.field_name)

    @field_validator("sample_rate", mode="after")
    @classmethod
    def _validate_sample_rate(cls, v, info):
        if v is None:
            return v
        return require_dimensionality(v, 1 / ureg.second, info.field_name)

    @field_validator("sample_width", mode="after")
    @classmethod
    def _validate_sample_width(cls, v, info):
        if v is None:
            return v
        return require_unit_family(v, ureg.byte, info.field_name, "data")

