"""Exercise image-layout contracts directly from the canonical source."""
from pathlib import Path
import runpy

import numpy as np
import pytest


@pytest.fixture(scope="module")
def source():
    return runpy.run_path(str(Path(__file__).resolve().parents[2] / "src/05_dataloader/05_dataloader.py"))


@pytest.mark.parametrize("layout,shape,width_axis,padded_shape", [
    ("HWC", (4, 8, 3), 1, (6, 10, 3)),
    ("CHW", (6, 8, 9), 2, (6, 10, 11)),
    ("HW", (4, 8), 1, (6, 10)),
])
def test_explicit_layout_preserves_channels(source, layout, shape, width_axis, padded_shape):
    x = np.arange(np.prod(shape)).reshape(shape)
    np.testing.assert_array_equal(source["RandomHorizontalFlip"](1, layout=layout)(x), np.flip(x, width_axis))
    padded = source["_pad_image"](x, 1, layout=layout)
    assert padded.shape == padded_shape
    crop_size = shape[:2] if layout == "HWC" else shape[-2:]
    np.testing.assert_array_equal(source["RandomCrop"](crop_size, padding=0, layout=layout)(x), x)
    tensor = source["Tensor"](x)
    result = source["RandomHorizontalFlip"](1, layout=layout)(tensor)
    assert isinstance(result, source["Tensor"])
    np.testing.assert_array_equal(result.data, np.flip(x, width_axis))


@pytest.mark.parametrize("shape", [(8, 9), (3, 8, 9), (8, 9, 3)])
def test_legacy_callers_retain_layout_inference(source, shape):
    x = np.arange(np.prod(shape)).reshape(shape)
    size = shape[:2] if len(shape) == 3 and shape[0] > 4 else shape[-2:]
    np.testing.assert_array_equal(source["RandomCrop"](size, padding=0)(x), x)


def test_layout_validation_is_independent_of_flip_probability(source):
    for p in (0, 1):
        with pytest.raises(ValueError, match="dimensions"):
            source["RandomHorizontalFlip"](p, layout="HWC")(np.ones((8, 8)))
    with pytest.raises(ValueError, match="layout"):
        source["RandomCrop"](4, layout="typo")
    with pytest.raises(ValueError, match="dimensions"):
        source["RandomCrop"](4, layout="HWC")(np.ones((8, 8)))
