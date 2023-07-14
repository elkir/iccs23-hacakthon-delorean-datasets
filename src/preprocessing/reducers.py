import re
import xarray as xr
import numpy as np

default_vars = ["t2m","d2m","stl4","ssrd","strd","w10","w100"]


# distribution indices
# Kotlarski, S. et al. (2014) ‘Regional climate modeling on European scales: A joint standard evaluation of the EURO-CORDEX RCM ensemble’, Geoscientific Model Development, 7(4), pp. 1297–1333. Available at: https://doi.org/10.5194/GMD-7-1297-2014.
def spatial_mean(ds, vars=default_vars, **kwargs):
    # weighted by area 
    return ds[vars].weighted(np.cos(np.deg2rad(ds.latitude))).mean(["latitude","longitude"])


def average_over_shape(da, shape):
    import shapely.vectorized
    x,y = np.meshgrid(da["longitude"], da["latitude"])
    # create a mask from the shape
    mask = shapely.vectorized.contains(shape, x,y)
    # average over the mask
    return da.where(mask).mean(dim=["latitude", "longitude"])




def calculate_climatological_spatial_mean(ds, vars=default_vars, **kwargs):
    return spatial_mean(ds,vars=vars).mean(["number"]) # step 

def calculate_variance(ds, vars=default_vars,start_time=None):
    """
    Calculate variance over ensembles, averaged over latitude and longitude, daily
    """
    ds_var = spatial_mean(ds[vars].resample(step="1D").mean().var(dim=["number"]))
    if start_time==None:
        start_time=ds_var.time
    ds_var = ds_var.assign_coords(valid_time=ds_var.step+start_time)
    return ds_var

def cross_date_reducer_wrapper(reducer, ds, vars=default_vars):
    """
    Wrapper for reducers that need to be applied to a cross-date dataset
    """
    # reduce over ensembles and files, but not over latitude and longitude
    return ds[vars].groupby("valid_time").reduce(reducer, dim=['number', 'time'])

def cross_date_mean(ds, vars=default_vars):
    """
    Average over ensembles and time, but not over latitude and longitude
    """
    return cross_date_reducer_wrapper('mean', ds, vars=vars)

def cross_date_variance(ds, vars=default_vars):
    """
    Average over ensembles and time, but not over latitude and longitude
    """
    return cross_date_reducer_wrapper('var', ds, vars=vars)
