import numpy as np
import matplotlib.pyplot as plt

data = np.fromfile(
    "Datasets/Max-Temp/Maxtemp_MaxT_2025.GRD",
    dtype=np.float32
)

# IMD missing value
data[data == 99.9] = np.nan

temp = data.reshape(365, 31, 31)

print("Shape:", temp.shape)

print("Min:", np.nanmin(temp))
print("Max:", np.nanmax(temp))
print("Mean:", np.nanmean(temp))

plt.figure(figsize=(8,6))

plt.imshow(temp[180])

plt.colorbar(label="Max Temperature (°C)")

plt.title("Day 180 Max Temperature")

plt.show()