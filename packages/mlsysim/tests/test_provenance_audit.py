import unittest

from mlsysim.tools.audit_provenance import (
    audit_datasets,
    audit_hardware_tech,
    audit_infra_capacity,
    audit_infra_facilities,
    audit_infra_grids,
    audit_infra_pricing,
    audit_literature_sourced,
    audit_ops_sourced,
    audit_platforms,
    audit_reference_stats,
    audit_registries,
    audit_systems_reference_values,
    audit_systems_topology,
    audit_systems_reliability,
)

class TestProvenanceAudit(unittest.TestCase):
    def test_all_registries_have_provenance(self):
        issues = audit_registries(scope_cloud=False)
        self.assertEqual(issues, [], "\n".join(issues))

    def test_hardware_tech_has_provenance(self):
        issues = audit_hardware_tech()
        self.assertEqual(issues, [], "\n".join(issues))

    def test_datasets_have_provenance(self):
        issues = audit_datasets()
        self.assertEqual(issues, [], "\n".join(issues))

    def test_platforms_have_provenance(self):
        issues = audit_platforms()
        self.assertEqual(issues, [], "\n".join(issues))

    def test_systems_topology_has_provenance(self):
        issues = audit_systems_topology()
        self.assertEqual(issues, [], "\n".join(issues))

    def test_systems_reference_values_have_provenance(self):
        issues = audit_systems_reference_values()
        self.assertEqual(issues, [], "\n".join(issues))

    def test_infra_grids_have_provenance(self):
        issues = audit_infra_grids()
        self.assertEqual(issues, [], "\n".join(issues))

    def test_infra_facilities_have_provenance(self):
        issues = audit_infra_facilities()
        self.assertEqual(issues, [], "\n".join(issues))

    def test_infra_pricing_have_provenance(self):
        issues = audit_infra_pricing()
        self.assertEqual(issues, [], "\n".join(issues))

    def test_infra_capacity_have_provenance(self):
        issues = audit_infra_capacity()
        self.assertEqual(issues, [], "\n".join(issues))

    def test_literature_sourced_have_provenance(self):
        issues = audit_literature_sourced()
        self.assertEqual(issues, [], "\n".join(issues))

    def test_reference_stats_have_provenance(self):
        issues = audit_reference_stats()
        self.assertEqual(issues, [], "\n".join(issues))

    def test_ops_sourced_have_provenance(self):
        issues = audit_ops_sourced()
        self.assertEqual(issues, [], "\n".join(issues))

    def test_systems_reliability_have_provenance(self):
        issues = audit_systems_reliability()
        self.assertEqual(issues, [], "\n".join(issues))

if __name__ == "__main__":
    unittest.main()
