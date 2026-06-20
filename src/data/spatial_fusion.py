import numpy as np
import pandas as pd

from src.data.climate_loaders import (
    load_rainfall,
    load_max_temp,
    load_min_temp,
    get_rainfall_coordinates,
    get_temperature_coordinates
)

YEAR = 2023

print("Loading datasets...")

rainfall = load_rainfall(YEAR)
max_temp = load_max_temp(YEAR)
min_temp = load_min_temp(YEAR)

rain_lats, rain_lons = get_rainfall_coordinates()
temp_lats, temp_lons = get_temperature_coordinates()

records = []

print("Building spatial fusion dataset...")

for day in range(rainfall.shape[0]):

    if day % 25 == 0:
        print(f"Processing Day {day + 1}")

    for lat_idx, temp_lat in enumerate(temp_lats):

        for lon_idx, temp_lon in enumerate(temp_lons):

            max_t = max_temp[day, lat_idx, lon_idx]
            min_t = min_temp[day, lat_idx, lon_idx]

            if np.isnan(max_t) or np.isnan(min_t):
                continue

            # ----------------------------------
            # Find rainfall cells inside
            # this 1° x 1° temperature cell
            # ----------------------------------

            lat_mask = (
                (rain_lats >= temp_lat - 0.5)
                &
                (rain_lats < temp_lat + 0.5)
            )

            lon_mask = (
                (rain_lons >= temp_lon - 0.5)
                &
                (rain_lons < temp_lon + 0.5)
            )

            rainfall_block = rainfall[
                day
            ][lat_mask][:, lon_mask]

            avg_rainfall = np.nanmean(rainfall_block)

            if np.isnan(avg_rainfall):
                continue

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

df.to_csv(
    output_file,
    index=False
)

print()
print(f"Saved -> {output_file}")