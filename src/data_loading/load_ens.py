# Data processing functions
import numpy as np
import xarray as xr

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s- Load - %(message)s')

def calculate_wind_speed(ds, drop_uv=True, verbose=False):
    """Calculate wind speed from u and v components and add it to the dataset

    Args:
        ds (xr.Dataset): Dataset containing u10, v10, u100, v100
        drop_uv (bool, optional): Drop u and v components from dataset. Defaults to True.

    Returns:
        xr.Dataset: Dataset with wind speed of 10m and 100m
        
        Calculate wind function
        returns a new dataset with the wind speed variable added,
        for both 10m and 100m, optionally (not) dropping the u and v components
    
        Copies the attributes from the u and v components to the wind speed components
        and replaces the non-shared attributes with the ones for the wind speed:
        
        #  index: w10, w100
        #  GRIB_cfVarName, GRIB_shortName : 10si, 100si
        #  long name,GRIB_name: 10m wind speed, 100m wind speed
        #  GRIB_paramId: 207, 228249

    """
    # check if any wind speed is in the dataset
    if ( "u10" not in ds and "v10" not in ds) and ( "u100" not in ds and "v100" not in ds):
        print ("No wind speed in dataset")
        return ds
    if "u10" in ds and "v10" in ds:
        ds = ds.assign(w10=np.sqrt(ds.u10**2 + ds.v10**2))
        ds.w10.attrs = ds.u10.attrs.copy()
        ds.w10.attrs.update({"GRIB_cfVarName": "10si", "GRIB_shortName": "10si", "long_name": "10m wind speed", "GRIB_name": "10m wind speed", "GRIB_paramId": 207})
        if drop_uv:
            ds = ds.drop(["u10", "v10"])
    # do if 100m wind speed is in the dataset
    if "u100" in ds and "v100" in ds:
        ds = ds.assign(w100=np.sqrt(ds.u100**2 + ds.v100**2))
        ds.w100.attrs = ds.u100.attrs.copy()
        ds.w100.attrs.update({"GRIB_cfVarName": "100si", "GRIB_shortName": "100si", "long_name": "100m wind speed", "GRIB_name": "100m wind speed", "GRIB_paramId": 228249})
        if drop_uv:
            ds = ds.drop(["u100", "v100"])
    return ds

def calculate_temperature_in_C(ds, verbose=False):
    """
    Calculate temperature in C from K and add it to the dataset
    
    Args:
        ds (xr.Dataset): Dataset containing t2m, d2m, stl4
        verbose (bool, optional): Print verbose output. Defaults to False.
    
    Returns:
        xr.Dataset: Dataset with temperature in C for t2m, d2m, stl4
        
        Calculate temperature function
        returns a new dataset with the temperature variable added,
        for t2m, d2m, stl4, optionally (not) dropping the K temperature components
        and updating the attributes to C
        
        #  index: t2m, d2m, stl4
        #  GRIB_cfVarName, GRIB_shortName : 2t, 2d, stl4
        #  long name,GRIB_name: 2m temperature, 2m dewpoint temperature, Soil temperature level 4
        #  GRIB_paramId: 167, 168, 235
    """
    # check if any temperature is in the dataset
    if ("t2m" not in ds) and ("d2m" not in ds) and ("stl4" not in ds):
        print ("No temperature in dataset")
    if "t2m" in ds:
        # subtract 273.15 to get temperature in C
        attrs = ds.t2m.attrs.copy()
        ds = ds.assign(t2m=ds.t2m - 273.15)
        # update the attributes
        # modify the units and GRIB_units to C
        attrs["units"] = "C"
        attrs["GRIB_units"] = "C"
        ds.t2m.attrs = attrs
    if "d2m" in ds:
        # subtract 273.15 to get temperature in C
        attrs = ds.d2m.attrs.copy()
        ds = ds.assign(d2m=ds.d2m - 273.15)
        # update the attributes
        # modify the units and GRIB_units to C
        attrs["units"] = "C"
        attrs["GRIB_units"] = "C"
        ds.d2m.attrs = attrs
    if "stl4" in ds:
        attrs = ds.stl4.attrs.copy()
        ds = ds.assign(stl4=ds.stl4 - 273.15)
        attrs["units"] = "C"
        attrs["GRIB_units"] = "C"
        ds.stl4.attrs = attrs
    return ds

# get diff values from monotonically increasing variables
def get_diff_values(ds, vars=["ssrd", "strd", "tp", "tcc", "ssr"], verbose=False):
    """Get diff values from monotonically increasing variables

    Args:
        ds (xarray.Dataset): Dataset containing the variables
        vars ( list , optional): Variables by index name. Defaults to ["ssrd", "strd", "tp", "tcc", "ssr"].
        verbose (bool, optional): Printing verbose messages. Defaults to False.

    Returns:
        xr.Dataset: Dataset with the diff values for the monotonically increasing variables
    """
    ds_step = ds.step.diff(dim="step") / np.timedelta64(1, 'h')
    def get_diff(da):
        # preserve the order of the dimensions
        original_dims = da.dims

        # do a diff on the xarray Dataset and add the first value to the beginning
        # to get the same shape as the original dataset
        ddiff = da.diff(dim="step")
        ddiff = ddiff / ds_step
        ddiff = xr.concat([0*da.isel(step=0), ddiff], dim="step") # prepend 0 to the beginning
        #TODO check if all atributes are correct
        ddiff = ddiff.transpose(*original_dims)
        ddiff.attrs = da.attrs.copy()
        ddiff.attrs["long_name"] = f"Δ {da.attrs['long_name']}"
        ddiff.attrs["units"] = f"{da.attrs['units']}/h"
        return ddiff
    
    for var in vars:
        if var in ds:
            ds = ds.assign({var: get_diff(ds[var])})
        else:
            if verbose:
                print(f"{var} not in dataset to calculate diff")
            
    return ds

# Load E and D data

def load_ens_data_ED(fn_E, fn_D, load_full_D=False,
                     drop_wind_components=True, temperature_in_C=True,
                     calculate_diffs=True,
                     verbose=False):
    logging.info(f"Loading {fn_E}")
    dsE = xr.load_dataset(fn_E, engine='cfgrib')

    if load_full_D:
        # loading everything is slower:
        # 51.2 s ± 6.44 s
        logging.info(f"Loading {fn_D} (full)")
        dsD = xr.load_dataset(fn_D, engine='cfgrib')     
    else:
        # filter variables out that are not needed
        # 6.15 s ± 454 ms per loop (mean ± std. dev. of 7 runs, 1 loop each) <- ERROR: this depends on file indexing
        logging.info(f"Loading {fn_D} (reduced)")
        l = [xr.load_dataset(fn_D, engine='cfgrib',
                            backend_kwargs={'filter_by_keys': {'number': i}})
                    for i in range(1,6)]
        # combine all the datasets into one along the number dimension
        dsD = xr.concat(l, dim="number")
        
    # =======================================
    ## Combine datasets, calculate wind
    # =========================================
    # check if the first step of dsD is the same as the last step of dsE across all variables,fields and (shared) ensembles
    assert (dsD.sel(number=dsE.number).isel(step=0) == dsE.isel(step=-1)).all()

    # combine the two datasets along the step dimension 
    ds = xr.concat([dsE[dsD.data_vars.keys()].isel(step=slice(None,-1)),
                    dsD.sel(number=dsE.number)],
                dim="step")
    
    # log which fields are processed based on the flags (wind, temperature, diffs)
    logging.info(f"Processing fields: {'wind ' if drop_wind_components else ''}"
             f"{'temperature ' if temperature_in_C else ''}"
             f"{'diffs ' if calculate_diffs else ''}"
             )
    ds = calculate_wind_speed(ds, drop_uv= drop_wind_components,verbose=verbose)
    dsD = calculate_wind_speed(dsD, drop_uv= drop_wind_components,verbose=verbose)
    if temperature_in_C:
        ds = calculate_temperature_in_C(ds)
        dsD = calculate_temperature_in_C(dsD)
    if calculate_diffs:
        ds = get_diff_values(ds,verbose=verbose)
        dsD = get_diff_values(dsD,verbose=verbose)

    # check if the values in step xr.DataArray are unique
    steps =(ds.step / np.timedelta64(1, 'D')).round(2)
    assert steps.size == np.unique(steps).size
    
    logging.info(f"Loading complete")
    return ds, dsD

def average_over_shape(da, shape):
    import shapely.vectorized
    x,y = np.meshgrid(da["longitude"], da["latitude"])
    # create a mask from the shape
    mask = shapely.vectorized.contains(shape, x,y)
    # average over the mask
    return da.where(mask).mean(dim=["latitude", "longitude"])

