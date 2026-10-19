import pandas as pd
ts = pd.to_datetime(["2021-09-17 00:00:00", "2021-09-17 01:00:00", "2021-09-17 02:00:00",
                     "2021-09-17 03:00:00", "2021-09-17 04:00:00", "2021-09-17 05:00:00"], utc=True)
pd.DataFrame({"event_timestamp": ts, "driver_id": [101] * 6,
              "trips_today": [0, 1, 3, 4, 6, 8],
              "earnings_today": [0.0, 50.0, 142.0, 184.0, 238.0, 301.0]}
             ).astype({"earnings_today": "float32"}).to_parquet("data/driver_hourly_stats.parquet")
