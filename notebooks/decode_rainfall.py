import numpy as np
import matplotlib.pyplot as plt

data = np.fromfile(
    "Datasets/Max-Temp/Rainfall_ind2025_rfp25.grd",
    dtype=np.float32
)

data[data == -999] = np.nan

rainfall = data.reshape(365,129,135)
latitudes = np.arange(6.5, 38.75, 0.25)
longitudes = np.arange(66.5, 100.25, 0.25)

# plt.figure(figsize=(10,6))
# plt.imshow(rainfall[180])
# plt.colorbar(label="Rainfall (mm)")
# plt.title("Day 180 Rainfall")
# plt.show()
print(len(latitudes))
print(len(longitudes))
print(latitudes[50])
print(longitudes[60])
