## Tutorial at [GRIB Data Example](https://docs.xarray.dev/en/stable/examples/ERA5-GRIB-example.html)


#%%  Import all the packages needed to explore grib data
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy

from pathlib  import Path
#%%
# wider plots
plt.rcParams['figure.figsize'] = [10, 5]

#%%
dir_data = Path('../data/ecmwf-ens')
fn_E = dir_data /"mars_v04e_2017-01-02_Mon.grib"
fn_D = dir_data /"mars_v04d_2017-01-02_Mon.grib"
#%% Load the data

ds = xr.load_dataset(fn_E, engine='cfgrib')
dsD = xr.load_dataset(fn_D, engine='cfgrib')
#%%

ds

#%%

# which variables are in ds but not in dsD
# difference between two sets
set(ds.data_vars.keys()) - set(dsD.data_vars.keys())

#%%

#print all the meta data for the temperature variable

# loop for all variables, print long name, units and 
for var in ds.data_vars:
    id = str(ds[var].GRIB_paramId)
    #check if id is 6 digits
    if len(id) == 6:
        id= id[3:6]+"."+id[0:3]
    else :
        id = id[0:3] + ".128"
    print(f"{id} - {var:4} - {ds[var].long_name} [{ds[var].units}]")

#%%
# select t2m variable, number 1, but all steps, latitudes and longitudes
t2m = ds.t2m[0] -273.15

#%%
t2m[0].plot()
#%%

fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.Robinson())
ax.coastlines(resolution="10m")
plot = t2m[0].plot(
    cmap=plt.cm.coolwarm, transform=ccrs.PlateCarree(), cbar_kwargs={"shrink": 0.6}
)
plt.title("ENS E1 - 2m temperature Europe January 2017")


#%%
t2m.sel(longitude=0, latitude=51.5, method="nearest").plot()
plt.title("ENS E1 - London 2m temperature January 2017")

# %%
# for lat 51.5 and lon 0, plot all numbers as lines
(ds.t2m -273.15).sel(longitude=0, latitude=51.5, method="nearest").plot.line( x="step")

# %%

# This line plot but turn off legend, all lines same colour and thinner
fig, ax = plt.subplots()
(dsD.t2m -273.15).sel(longitude=0, latitude=51.5, method="nearest").plot.line( x="step", ax=ax, color="k", linewidth=0.5, alpha=0.5)
ax.legend().set_visible(False)
# first week only
ax.set_xlim(1.5e15, 3e15)
# convert xaxis  from seconds to days 


