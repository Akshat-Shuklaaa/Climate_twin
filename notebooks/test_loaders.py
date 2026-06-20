from src.data.climate_loaders import *

rain = load_rainfall(2025)
max_t = load_max_temp(2025)
min_t = load_min_temp(2025)

print("Rainfall:", rain.shape)
print("Max Temp:", max_t.shape)
print("Min Temp:", min_t.shape)

print()

print("Rainfall Mean:", rain.mean())
print("Max Temp Mean:", max_t.mean())
print("Min Temp Mean:", min_t.mean())