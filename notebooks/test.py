from src.data.climate_loaders import *

lats, lons = get_temperature_coordinates()

print(len(lats))
print(len(lons))

print(lats[0], lats[-1])
print(lons[0], lons[-1])