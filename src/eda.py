# %%
import xarray as xr
from data_loading.load_ens import *

# ds = xr.open_mfdataset('../data/mars_v05e_*.grib', engine='cfgrib', concat_dim='time', combine='nested', parallel=True, chunks={'time': 3})
# ds = xr.open_dataset('../data/mars_v05e_2017-01-09_Mon.grib', engine='cfgrib')
# ds = calculate_wind_speed(ds)
# ds = calculate_temperature_in_C(ds)
# ds = get_diff_values(ds)

# use Dask to open multiple E files for distributed computing (1 chunk per file)
ds, dsD = load_multiple_ens_data_ED('../data')

# %%
def calculate_wind_power(speed, C=0.612):
    return C * speed**3  

# %%
# calculate wind power
ds = ds.assign(p10=calculate_wind_power(ds['w10']),
               p100=calculate_wind_power(ds['w100']))
ds

# %%
# reduce the dataset to time series of spatial and ensemble means
from preprocessing.reducers import calculate_climatological_spatial_mean
time_series = calculate_climatological_spatial_mean(ds)
time_series

# %%
