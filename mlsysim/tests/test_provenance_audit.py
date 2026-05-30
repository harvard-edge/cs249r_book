import unittest

from mlsysim.tools.appendix_lineage import (
    audit_appendix_defaults,
    audit_appendix_literature,
    audit_appendix_pricing,
    audit_appendix_reliability,
)
from mlsysim.tools.audit_provenance import (
    audit_infra_capacity,
    audit_infra_grids,
    audit_infra_pricing,
    audit_literature_sourced,
    audit_registries,
    audit_systems_reliability,
)

class TestProvenanceAudit(unittest.TestCase):
    def test_all_registries_have_provenance(self):
        issues = audit_registries(scope_cloud=False)
        self.assertEqual(issues, [], "\n".join(issues))

    def test_infra_grids_have_provenance(self):
        issues = audit_infra_grids()
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

    def test_systems_reliability_have_provenance(self):
        issues = audit_systems_reliability()
        self.assertEqual(issues, [], "\n".join(issues))

    def test_appendix_has_no_defaults_refs(self):
        issues = audit_appendix_defaults()
        self.assertEqual(issues, [], "\n".join(issues))

    def test_appendix_pricing_lineage(self):
        issues = audit_appendix_pricing()
        self.assertEqual(issues, [], "\n".join(issues))

    def test_appendix_reliability_lineage(self):
        issues = audit_appendix_reliability()
        self.assertEqual(issues, [], "\n".join(issues))

    def test_appendix_literature_lineage(self):
        issues = audit_appendix_literature()
        self.assertEqual(issues, [], "\n".join(issues))

if __name__ == "__main__":
    unittest.main()
