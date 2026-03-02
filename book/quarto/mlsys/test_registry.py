# test_registry.py
# Unit tests for the mlsysim Hardware and Models registries.

import unittest
from mlsysim import Hardware, Models
from mlsysim.constants import ureg

class TestMLSysRegistry(unittest.TestCase):
    def test_hardware_ridge_points(self):
        """Test that ridge points are calculated correctly and are positive."""
        h100 = Hardware.H100
        ridge = h100.ridge_point()
        self.assertGreater(ridge.magnitude, 0)
        self.assertEqual(ridge.units, ureg.parse_units('flop/byte'))
        
        # H100: ~2 PFLOPS / 3.35 TB/s = ~590 FLOP/byte
        self.assertGreater(ridge.magnitude, 100)
        self.assertLess(ridge.magnitude, 1000)

    def test_model_size(self):
        """Test model weight storage calculations."""
        gpt3 = Models.GPT3
        from mlsysim.constants import BYTES_FP16, BYTES_INT4
        
        # GPT-3 175B @ FP16 (2 bytes) = 350 GB
        size_fp16 = gpt3.size_in_bytes(BYTES_FP16)
        self.assertAlmostEqual(size_fp16.to('GB').magnitude, 350, delta=1)
        
        # GPT-3 175B @ INT4 (0.5 bytes) = 87.5 GB
        size_int4 = gpt3.size_in_bytes(BYTES_INT4)
        self.assertAlmostEqual(size_int4.to('GB').magnitude, 87.5, delta=1)

    def test_assertions(self):
        """Test that unrealistic hardware/models trigger assertions."""
        from mlsysim.hardware import HardwareSpec
        from mlsysim.models import ModelSpec
        
        # Non-positive bandwidth
        with self.assertRaises(AssertionError):
            HardwareSpec("Broken", 2024, 0 * ureg.GB/ureg.s, 1 * ureg.TFLOPs/ureg.s, 1 * ureg.GB)
            
        # Non-positive params
        with self.assertRaises(AssertionError):
            ModelSpec("Ghost", 0 * ureg.count, "Transformer")

if __name__ == '__main__':
    unittest.main()
