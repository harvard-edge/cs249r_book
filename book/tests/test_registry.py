# test_registry.py
# Unit tests for the mlsysim Hardware and Models registries.

import unittest
from mlsysim import Hardware, Models
from mlsysim.core.constants import ureg

class TestMLSysRegistry(unittest.TestCase):
    def test_hardware_ridge_points(self):
        """Test that ridge points are calculated correctly and are positive."""
        h100 = Hardware.H100
        ridge = h100.ridge_point()
        self.assertGreater(ridge.magnitude, 0)
        self.assertIn('flop', str(ridge.units))
        self.assertIn('B', str(ridge.units))
        
        # H100: ~2 PFLOPS / 3.35 TB/s = ~590 FLOP/byte
        self.assertGreater(ridge.magnitude, 100)
        self.assertLess(ridge.magnitude, 1000)

    def test_model_size(self):
        """Test model weight storage calculations."""
        gpt3 = Models.GPT3
        from mlsysim.core.constants import BYTES_FP16, BYTES_INT4
        
        # GPT-3 175B @ FP16 (2 bytes) = 350 GB
        size_fp16 = gpt3.size_in_bytes(BYTES_FP16)
        self.assertAlmostEqual(size_fp16.to('GB').magnitude, 350, delta=1)
        
        # GPT-3 175B @ INT4 (0.5 bytes) = 87.5 GB
        size_int4 = gpt3.size_in_bytes(BYTES_INT4)
        self.assertAlmostEqual(size_int4.to('GB').magnitude, 87.5, delta=1)

    def test_assertions(self):
        """Test that unrealistic hardware/models trigger assertions."""
        from mlsysim.hardware.types import HardwareNode
        from pydantic import ValidationError
        
        # Non-positive bandwidth
        with self.assertRaises(ValidationError):
            HardwareNode(name="Broken", release_year=2024, compute={"peak_flops": "not a number"}, memory={"capacity": "1 GB", "bandwidth": "0 GB/s"})

if __name__ == '__main__':
    unittest.main()
