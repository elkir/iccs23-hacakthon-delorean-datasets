import re
import xarray as xr
import numpy as np

default_vars = ["t2m","d2m","stl4","ssrd","strd","w10","w100"]

def calculate_variance(ds, vars=default_vars,start_time=None):
    """
    Calculate variance over ensembles, averaged over latitude and longitude, daily
    """
    ds_var = ds[vars].resample(step="1D").mean().var(dim=["number"]
                        ).mean(["latitude","longitude"])
    if start_time==None:
        start_time=ds_var.time
    ds_var = ds_var.assign_coords(valid_time=ds_var.step+start_time)
    return ds_var

# distribution indices
# Kotlarski, S. et al. (2014) ‘Regional climate modeling on European scales: A joint standard evaluation of the EURO-CORDEX RCM ensemble’, Geoscientific Model Development, 7(4), pp. 1297–1333. Available at: https://doi.org/10.5194/GMD-7-1297-2014.
def spatial_mean(ds, vars=default_vars, **kwargs):
    # weighted by area 
    return ds[vars].weighted(np.cos(np.deg2rad(ds.latitude))).mean(["latitude","longitude"])

def calculate_climatological_spatial_mean(ds, vars=default_vars, **kwargs):
    return spatial_mean(ds,vars=vars).mean(["number"]) # step 

