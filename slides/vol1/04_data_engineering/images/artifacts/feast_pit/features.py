# Feature view exactly as in the Feast docs, "Point-in-time joins" (v0.67.0);
# only the source path and timestamp field name are filled in.
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64
from datetime import timedelta

driver = Entity(name="driver", join_keys=["driver_id"])

driver_stats_fv = FeatureView(
    name="driver_hourly_stats",
    entities=[driver],
    schema=[
        Field(name="trips_today", dtype=Int64),
        Field(name="earnings_today", dtype=Float32),
    ],
    ttl=timedelta(hours=2),
    source=FileSource(
        path="data/driver_hourly_stats.parquet",
        timestamp_field="event_timestamp",
    )
)
