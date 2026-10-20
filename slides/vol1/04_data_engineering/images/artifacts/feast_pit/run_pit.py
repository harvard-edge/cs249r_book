"""Point-in-time join with Feast, the driver example from the Feast docs.

Docs: docs.feast.dev, "Point-in-time joins" (feast-dev/feast v0.67.0,
docs/getting-started/concepts/point-in-time-joins.md; Apache 2.0). Feature
rows and entity rows are the docs' own tables. Tested with feast 0.67.0.

Run from this folder:
  python3 make_data.py && feast apply && python3 run_pit.py
Printed with feast 0.67.0 (local file offline store): three rows. The two
entity rows with no feature value inside the 2-hour TTL are dropped, where
the docs' table shows them with NULL features.
"""
import pandas as pd
from feast import FeatureStore
store = FeatureStore(repo_path=".")
entity_df = pd.DataFrame({
    "event_timestamp": pd.to_datetime(["2021-09-16 23:00:00", "2021-09-17 01:15:30",
        "2021-09-17 04:32:45", "2021-09-17 05:43:10", "2021-09-17 16:26:17"], utc=True),
    "driver_id": [101] * 5,
    "trip_success": [0, 1, 0, 1, 0]})
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        'driver_hourly_stats:trips_today',
        'driver_hourly_stats:earnings_today'
    ],
).to_df()
pd.set_option("display.width", 200)
print(training_df.sort_values("event_timestamp").to_string(index=False))
