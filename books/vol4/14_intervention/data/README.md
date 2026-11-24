# Data Provenance (`14_intervention`)

## `ca_dmv_disengagements.csv`

- **Source**: State of California Department of Motor Vehicles, Autonomous Vehicle Disengagement Reports (`@cadmv2026disengagement`), archived bundle from <https://web.archive.org/web/20260519/https://www.dmv.ca.gov/portal/vehicle-industry-services/autonomous-vehicles/disengagement-reports/>.
- **Epistemic category**: `measured` (mandatory annual autonomous-vehicle testing filings under California Code of Regulations Title 13, Section 227.50).
- **Schema**: `company`, `report_year`, `reporting_period`, `permit_type` (`drivered_AVT` or `driverless_AVT`), `filer_type`, `autonomous_miles`, `disengagements`, `miles_per_disengagement`, `source_method`.
- **Accounting boundary**: Report years run December 1 through November 30. Years 2021–2025 sum per-VIN rows across all permittees in the DMV consolidated CSVs; years 2019–2020 cover all spreadsheet filers plus PDF filings for Waymo, Cruise, and Zoox; years 2015–2018 transcribe PDF filings for Waymo, Cruise, and Zoox. Commercial deployment miles under deployment permits are excluded by regulation, and disengagement reporting thresholds are self-defined by each permittee.
