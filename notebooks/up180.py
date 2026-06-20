import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Load GRD File
# =========================

data = np.fromfile(
    "Rainfall_ind2025_rfp25.grd",
    dtype=np.float32
)

# Replace missing values
data[data == -999] = np.nan

# Reshape
rainfall = data.reshape(365, 129, 135)

# =========================
# Create Latitude/Longitude Arrays
# =========================

latitudes = np.arange(6.5, 38.75, 0.25)
longitudes = np.arange(66.5, 100.25, 0.25)

# =========================
# Extract One Day
# =========================

day = 180

records = []

for lat_idx in range(129):

    for lon_idx in range(135):

        value = rainfall[day, lat_idx, lon_idx]

        if not np.isnan(value):

            records.append([
                latitudes[lat_idx],
                longitudes[lon_idx],
                value
            ])

df = pd.DataFrame(
    records,
    columns=[
        "latitude",
        "longitude",
        "rainfall"
    ]
)

# =========================
# Uttar Pradesh Filter
# =========================

up = df[
    (df["latitude"] >= 24) &
    (df["latitude"] <= 31) &
    (df["longitude"] >= 77) &
    (df["longitude"] <= 85)
]

# =========================
# Statistics
# =========================

print("\nUP DATA SHAPE")
print(up.shape)

print("\nFIRST 5 ROWS")
print(up.head())

print("\nAVERAGE RAINFALL")
print(up["rainfall"].mean())

print("\nMAX RAINFALL")
print(up["rainfall"].max())

print("\nTOTAL RAINFALL")
print(up["rainfall"].sum())

# =========================
# Plot UP Rainfall
# =========================

plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    up["longitude"],
    up["latitude"],
    c=up["rainfall"],
    cmap="Blues",
    s=40
)

plt.colorbar(scatter, label="Rainfall (mm)")

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.title("Uttar Pradesh Rainfall - Day 180")

plt.grid(True)

plt.show()