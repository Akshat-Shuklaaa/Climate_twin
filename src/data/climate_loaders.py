import numpy as np
def is_leap_year(year):
    return year % 4 == 0 and (
        year % 100 != 0 or year % 400 == 0
    )

# =====================================================
# RAINFALL LOADER
# Shape: (365, 129, 135)
# Missing value: -999
# =====================================================

def load_rainfall(year):
    days = 366 if is_leap_year(year) else 365
    filepath = f"Datasets/Rainfall/Rainfall_ind{year}_rfp25.grd"

    data = np.fromfile(
        filepath,
        dtype=np.float32
    )

    data[data == -999] = np.nan

    rainfall = data.reshape(days, 129, 135)

    return rainfall


# =====================================================
# MAX TEMPERATURE LOADER
# Shape: (365, 31, 31)
# Missing value: 99.9
# =====================================================

def load_max_temp(year):

    filepath = f"Datasets/Max-Temp/Maxtemp_MaxT_{year}.GRD"

    data = np.fromfile(
        filepath,
        dtype=np.float32
    )

    data[data == 99.9] = np.nan

    days = 366 if is_leap_year(year) else 365
    max_temp = data.reshape(days, 31, 31)

    return max_temp


# =====================================================
# MIN TEMPERATURE LOADER
# Shape: (365, 31, 31)
# Missing value: 99.9
# =====================================================

def load_min_temp(year):

    filepath = f"Datasets/Min_Temp/Mintemp_MinT_{year}.GRD"

    data = np.fromfile(
        filepath,
        dtype=np.float32
    )

    data[data == 99.9] = np.nan

    days = 366 if is_leap_year(year) else 365
    min_temp = data.reshape(days, 31, 31)

    return min_temp


def get_rainfall_coordinates():

    latitudes = np.arange(6.5, 38.75, 0.25)
    longitudes = np.arange(66.5, 100.25, 0.25)

    return latitudes, longitudes


def get_temperature_coordinates():

    latitudes = np.arange(7.5, 38.5, 1.0)
    longitudes = np.arange(67.5, 98.5, 1.0)

    return latitudes, longitudes