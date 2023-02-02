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
dir_data = Path('../ecmwf-ens')
fn_E = dir_data /"mars_v04e_2017-01-02_Mon.grib"
fn_D = dir_data /"mars_v04d_2017-01-02_Mon.grib"
#%% Load the data

ds = xr.load_dataset(fn_E, engine='cfgrib')
# dsD = xr.load_dataset(fn_D, engine='cfgrib')
#%%

ds

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



# %% 
# 
#Calculate correlation betweem ensembles (number field ) in ds.t2m
correlation = xr.corr(ds.t2m, ds.t2m, dim="number")

# %%
# extract number 1 and 2 of t2m into two different data arrays
# remove flat coordinates
#  smaller region
xval = ds.t2m[0].drop_vars(["number"])
yval = ds.t2m[1].drop_vars(["number"])
xval
# dataset = xr.Dataset({"number1": xval, "number2": yval, "step": ds.step})
# dataset.plot.scatter(x="number1", y="number2", c="step", cmap="viridis",  figsize=(10, 10))

# %%
