import unittest

from mlsysim.core.provenance import ProvenanceKind, Sourced
from mlsysim.hardware.registry import Hardware
from mlsysim.infrastructure.registry import Infrastructure
from mlsysim.systems.reliability import Reliability

class TestProvenance(unittest.TestCase):
    def test_hardware_cloud_has_provenance(self):
        h100 = Hardware.Cloud.H100
        prov = h100.metadata.provenance
        self.assertIsNotNone(prov)
        self.assertTrue(bool(prov.ref.strip()))

    def test_grid_has_provenance(self):
        grid = Infrastructure.Grids.US_Avg
        prov = grid.metadata.provenance
        self.assertIsNotNone(prov)

    def test_reliability_mttf_is_sourced(self):
        mttf = Reliability.Gpu.mttf_hours
        self.assertIsInstance(mttf, Sourced)
        prov = mttf.provenance
        self.assertIsNotNone(prov)
        self.assertEqual(prov.kind, ProvenanceKind.LITERATURE)

    def test_defaults_module_removed(self):
        import importlib.util

        self.assertIsNone(importlib.util.find_spec("mlsysim.core.defaults"))

    def test_no_flat_registry_aliases_at_package_root(self):
        import mlsysim

        for name in ("GPUS_PER_HOST", "ALLREDUCE_FACTOR", "GPU_MTTF_HOURS"):
            self.assertFalse(
                hasattr(mlsysim, name),
                f"remove package-root alias {name}; use Systems/Literature registries",
            )

if __name__ == "__main__":
    unittest.main()
