# %%
import xarray as xr
from data_loading.load_ens import *

# use Dask to open multiple E files for distributed computing (1 chunk per file)
ds = xr.open_mfdataset('../data/mars_v05e_*.grib', engine='cfgrib', concat_dim='time', combine='nested', parallel=True, chunks={'time': 3})
ds

# %%
ds = calculate_wind_speed(ds)
ds = calculate_temperature_in_C(ds)
ds = get_diff_values(ds)

# %%
def calculate_wind_power(speed, C=0.612):
    return C * speed**3  

# %%
# calculate wind power
ds = ds.assign(p10=calculate_wind_power(ds['w10']),
               p100=calculate_wind_power(ds['w100']))
ds

# %%
# reduce the dataset to time series of spatial means
from preprocessing.reducers import spatial_mean
ds_reduced = spatial_mean(ds)
ds_reduced

# %%
