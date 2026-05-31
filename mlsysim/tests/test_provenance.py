import unittest

from pydantic import ValidationError

from mlsysim.core.provenance import Provenance, ProvenanceKind, Sourced
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

    def test_provenance_requires_verified_date(self):
        with self.assertRaises(ValidationError):
            Provenance(kind=ProvenanceKind.CONVENTION, ref="missing date")

    def test_evidence_provenance_requires_url(self):
        with self.assertRaises(ValidationError):
            Provenance(
                kind=ProvenanceKind.LITERATURE,
                ref="paper without URL",
                verified="2026-05-31",
            )

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

    def test_no_flat_model_aliases(self):
        from mlsysim import Models

        for name in ("GPT3", "ResNet50", "MobileNetV2"):
            self.assertFalse(
                hasattr(Models, name),
                f"remove Models.{name}; use the category namespace",
            )

    def test_gpt3_training_flops_owned_by_model_registry(self):
        from mlsysim import Literature, Models

        self.assertFalse(hasattr(Literature.Benchmarks, "GPT3TrainingFlops"))
        self.assertIsNotNone(Models.Language.GPT3.training_ops)

if __name__ == "__main__":
    unittest.main()
