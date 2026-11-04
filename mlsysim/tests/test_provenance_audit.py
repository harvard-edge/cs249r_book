import unittest

from mlsysim.tools.audit_provenance import (
    audit_actuators,
    audit_agents,
    audit_datasets,
    audit_embodied,
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
    audit_sensors,
    audit_systems_reference_values,
    audit_systems_topology,
    audit_systems_reliability,
    audit_tier2_book_series,
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

    def test_sensors_have_provenance(self):
        issues = audit_sensors()
        self.assertEqual(issues, [], "\n".join(issues))

    def test_actuators_have_provenance(self):
        issues = audit_actuators()
        self.assertEqual(issues, [], "\n".join(issues))

    def test_embodied_have_provenance(self):
        issues = audit_embodied()
        self.assertEqual(issues, [], "\n".join(issues))

    def test_agents_have_provenance(self):
        issues = audit_agents()
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

    def test_tier2_book_series_have_provenance(self):
        issues = audit_tier2_book_series()
        self.assertEqual(issues, [], "\n".join(issues))

    def test_series_provenance_parser_and_figure_checker_catch_malformed_headers(self):
        import sys
        import tempfile
        from pathlib import Path
        from mlsysim.core.series_loader import LocalSeries

        repo_root = Path(__file__).resolve().parents[2]
        binder_dir = repo_root / "binder"
        if str(binder_dir) not in sys.path:
            sys.path.insert(0, str(binder_dir))
        from cli.checks import figure_data_provenance as fdp

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            bad_csv = tmp_dir / "bad.csv"
            bad_csv.write_text(
                "# Verified: Oct 2026\n"
                "# Epistemic: empirical\n"
                "# Upstream-SHA256: deadbeef\n"
                "# Generator: python3 binder/tools/scripts/does_not_exist.py\n"
                "x,y\n1,2\n",
                encoding="utf-8",
            )
            violations = fdp.check_csv_file(bad_csv, repo_root=repo_root, bib_cache={None: {"patterson2021carbon"}})
            codes = {v.code for v in violations}
            self.assertEqual(
                codes,
                {
                    "series_missing_source",
                    "series_invalid_verified",
                    "series_missing_computation_or_notes",
                    "series_invalid_epistemic",
                    "series_invalid_sha256",
                    "series_missing_generator_script",
                },
            )

            good_csv = tmp_dir / "good.csv"
            good_csv.write_text(
                "# Source: Patterson et al. (2021), Table 4 (@patterson2021carbon)\n"
                "# Verified: 2026-10-20\n"
                "# Epistemic: measured (metered system power * PUE)\n"
                "# Computation: E_MWh = chips * kw * hours / 1000\n"
                "x,y\n1,2\n",
                encoding="utf-8",
            )
            series = LocalSeries(good_csv)
            self.assertEqual(series.provenance["verified"], "2026-10-20")
            self.assertTrue(series.provenance["epistemic"].startswith("measured"))
            self.assertEqual(
                fdp.check_csv_file(good_csv, repo_root=repo_root, bib_cache={None: {"patterson2021carbon"}}),
                [],
            )

            bad_qmd = tmp_dir / "chapter.qmd"
            bad_qmd.write_text(
                '::: {#fig-demo fig-cap="**Demo**: Test."}\n```{python}\nimport matplotlib.pyplot as plt\n```\n:::\n',
                encoding="utf-8",
            )
            checked, cell_violations = fdp.collect_cell_provenance([bad_qmd], repo_root=repo_root)
            self.assertEqual(checked, 1)
            self.assertEqual([v.code for v in cell_violations], ["figure_missing_cell_provenance"])

if __name__ == "__main__":
    unittest.main()
