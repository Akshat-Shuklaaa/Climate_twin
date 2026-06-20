import numpy as np
import pandas as pd

from src.data.climate_loaders import (
    load_rainfall,
    load_max_temp,
    load_min_temp,
    get_temperature_coordinates
)

YEAR = 2025

print("Loading datasets...")

rainfall = load_rainfall(YEAR)
max_temp = load_max_temp(YEAR)
min_temp = load_min_temp(YEAR)

temp_lats, temp_lons = get_temperature_coordinates()

records = []

print("Building climate dataset...")

for day in range(365):

    for lat_idx in range(31):

        for lon_idx in range(31):

            max_t = max_temp[day, lat_idx, lon_idx]
            min_t = min_temp[day, lat_idx, lon_idx]

            if np.isnan(max_t) or np.isnan(min_t):
                continue

            temp_lat = temp_lats[lat_idx]
            temp_lon = temp_lons[lon_idx]

            # -------------------------------------------------
            # Convert 1° temp cell to rainfall indices
            # -------------------------------------------------

            rain_lat_start = lat_idx * 4
            rain_lat_end   = rain_lat_start + 4

            rain_lon_start = lon_idx * 4
            rain_lon_end   = rain_lon_start + 4

            rainfall_block = rainfall[
                day,
                rain_lat_start:rain_lat_end,
                rain_lon_start:rain_lon_end
            ]

            avg_rainfall = np.nanmean(rainfall_block)

            records.append([
                day + 1,
                temp_lat,
                temp_lon,
                avg_rainfall,
                max_t,
                min_t
            ])

print("Creating dataframe...")

df = pd.DataFrame(
    records,
    columns=[
        "day",
        "latitude",
        "longitude",
        "rainfall",
        "max_temp",
        "min_temp"
    ]
)

print(df.head())
print()
print("Rows:", len(df))

output_file = f"climate_{YEAR}.csv"

df.to_csv(output_file, index=False)

print()
print(f"Saved: {output_file}")