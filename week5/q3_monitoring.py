import pandas as pd
from evidently.report import Report
from evidently.metrics import ColumnQuantileMetric
from datetime import datetime

# Load data
df = pd.read_parquet("data/green_tripdata_2024-03.parquet")
df["lpep_pickup_datetime"] = pd.to_datetime(df["lpep_pickup_datetime"])

# Reference = March 1–7
reference_data = df[df["lpep_pickup_datetime"] < "2024-03-08"]

# Store median values
daily_quantiles = {}

# Loop March 8–31
for day in range(8, 32):
    date_str = f"2024-03-{day:02d}"
    current_data = df[df["lpep_pickup_datetime"].dt.date == pd.to_datetime(date_str).date()]

    if current_data.empty:
        print(f"{date_str}: ⚠️ No data found")
        continue

    if "fare_amount" not in current_data.columns:
        print(f"{date_str}: ❌ 'fare_amount' column missing")
        continue

    if current_data["fare_amount"].isna().all():
        print(f"{date_str}: ⚠️ All fare_amount values are NaN")
        continue

    report = Report(metrics=[
        ColumnQuantileMetric(column_name="fare_amount", quantile=0.5)
    ])

    try:
        report.run(reference_data=reference_data, current_data=current_data)
        result = report.as_dict()

        metric_result = result["metrics"][0]["result"]

        # Check for the right nested structure
        if "current" in metric_result and "value" in metric_result["current"]:
            median = metric_result["current"]["value"]
            daily_quantiles[date_str] = median
            print(f"{date_str}: ✅ median fare_amount = {median}")
        else:
            print(f"{date_str}: ❌ Unexpected metric format, 'current.value' missing")

    except Exception as e:
        print(f"{date_str}: ❌ Error running report: {e}")

# Final max result
if daily_quantiles:
    max_day = max(daily_quantiles, key=daily_quantiles.get)
    max_value = daily_quantiles[max_day]
    print("\n✅ Maximum fare_amount quantile=0.5:")
    print(f"{max_day}: {max_value}")
else:
    print("\n❌ No valid median values computed.")
