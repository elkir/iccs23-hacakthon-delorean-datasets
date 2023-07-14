# Data processing functions
import numpy as np
import xarray as xr

import logging
from pathlib import Path
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


# Load E and D data - match the 5 ensemble members between E and D

def load_ens_data_ED(fn_E, fn_D, load_full_D=False,
                     drop_wind_components=True, temperature_in_C=True,
                     calculate_diffs=True,
                     verbose=False):
    """Load data from both the E and the D file - match the 5 ensemble members between E and D

    Args:
        fn_E (str): Filename of the E file
        fn_D (str): Filename of the D file
        load_full_D (bool, optional): Whether to load all 50 ensembles to the D dataset or only the five matching the ones on file E. Defaults to False.
        drop_wind_components (bool, optional): Drop the U and V components of wind when computing speed. Defaults to True.
        temperature_in_C (bool, optional): Convert temperature fields to Celsius. Defaults to True.
        calculate_diffs (bool, optional): Calculate Δ fields for solar radiation . Defaults to True.
        verbose (bool, optional): Print all detials of processing . Defaults to False.

    Returns:
        ds (xr.Dataset):  Dataset with the 5 ensemble members from both E and D files continous in time for the 6 weeks
        dsD (xr.Dataset): Dataset with the 5 or 50 ensemble members from the D file. Weeks 2-6.
    """
    
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
        with xr.open_dataset(fn_D, engine='cfgrib') as dsD:
            dsD = dsD.sel(number=dsE.number).load()

    # =======================================
    ## Combine datasets, calculate wind
    # =========================================
    # check if the first step of dsD is the same as the last step of dsE across all variables,fields and (shared) ensembles
    assert (dsD.sel(number=dsE.number).isel(step=0) == dsE.isel(step=-1)).all()

    # combine the two datasets along the step dimension 
    ds = xr.concat([dsE[dsD.data_vars.keys()].isel(step=slice(None,-1)),
                    dsD.sel(number=dsE.number)], dim="step")
    
    ds = preprocess(ds, drop_wind_components=drop_wind_components,
                    temperature_in_C=temperature_in_C,
                    calculate_diffs=calculate_diffs,
                    verbose=verbose)
    dsD = preprocess(dsD, drop_wind_components=drop_wind_components,
                    temperature_in_C=temperature_in_C,
                    calculate_diffs=calculate_diffs,
                    verbose=verbose)
    
    # check if the values in step xr.DataArray are unique
    steps =(ds.step / np.timedelta64(1, 'D')).round(2)
    assert steps.size == np.unique(steps).size
    
    logging.info(f"Loading complete")
    return ds, dsD

# load only the 50 ensemble members from D
def load_ens_data_D(fn_D,
                    drop_wind_components=True, temperature_in_C=True,
                    calculate_diffs=True,
                    verbose=False):
    """Load the 50 ensembles from the D file

    Args:
        fn_D (str): Filename of the D file
        load_full_D (bool, optional): Whether to load all 50 ensembles to the D dataset or only the five matching the ones on file E. Defaults to False.
        drop_wind_components (bool, optional): Drop the U and V components of wind when computing speed. Defaults to True.
        temperature_in_C (bool, optional): Convert temperature fields to Celsius. Defaults to True.
        calculate_diffs (bool, optional): Calculate Δ fields for solar radiation . Defaults to True.
        verbose (bool, optional): Print all detials of processing . Defaults to False.
    Returns:
        dsD (xr.Dataset): Dataset with the 50 ensemble members from the D file. Weeks 2-6.
    """
    # loading everything is slower:
    # 51.2 s ± 6.44 s
    logging.info(f"Loading {fn_D} (full)")
    dsD = xr.load_dataset(fn_D, engine='cfgrib')
    dsD = preprocess(dsD, drop_wind_components=drop_wind_components,
                     temperature_in_C=temperature_in_C,
                     calculate_diffs=calculate_diffs,
                     verbose=verbose)
    logging.info(f"Loading complete")
    return dsD

def preprocess(ds, drop_wind_components=True, temperature_in_C=True,
               calculate_diffs=True, verbose=False):
    """Preprocess the dataset.
    
    Args:
        ds (xr.Dataset): The dataset.
        drop_wind_components (bool, optional): Drop the U and V components of wind when computing speed. Defaults to True.
        temperature_in_C (bool, optional): Convert temperature fields to Celsius. Defaults to True.
        calculate_diffs (bool, optional): Calculate Δ fields for solar radiation . Defaults to True.
        verbose (bool, optional): Print all detials of processing . Defaults to False.
    Returns:
        ds (xr.Dataset): Preprocessed dataset.
        dsD (xr.Dataset): Dataset with the 5 or 50 ensemble members from the D file.
    """
    # log which fields are processed based on the flags (wind, temperature, diffs)
    logging.info(f"Processing fields: {'wind ' if drop_wind_components else ''}"
                 f"{'temperature ' if temperature_in_C else ''}"
                 f"{'diffs ' if calculate_diffs else ''}")
    ds = calculate_wind_speed(ds, drop_uv= drop_wind_components,verbose=verbose)
    if temperature_in_C:
        ds = calculate_temperature_in_C(ds)
    if calculate_diffs:
        ds = get_diff_values(ds,verbose=verbose)
    return ds
 

def load_multiple_ens_data_ED(dir_or_files, 
                              load_full_D=False,
                              drop_wind_components=True,
                              temperature_in_C=True,
                              calculate_diffs=True,
                              verbose=False,
                              version='v05',
                              chunks={'time': 1}):
    if type(dir_or_files) is str:
        dir = Path(dir_or_files)
        e_files = sorted(dir.glob(f'mars_{version}_*.grib'))
        d_files = sorted(dir.glob(f'mars_{version}_*.grib'))
    else:
        e_files = dir_or_files
        d_files = [f.replace(f"_{version}e_", f"_{version}d_") for f in e_files]
    
    dsE = xr.open_mfdataset(e_files, engine='cfgrib',
                            concat_dim='time', combine='nested',
                            parallel=True, chunks=chunks)
    dsD = xr.open_mfdataset(d_files, engine='cfgrib',
                            concat_dim='time', combine='nested',
                            parallel=True, chunks=chunks)

    if not load_full_D:
        dsD = dsD.sel(number=dsE.number)

    assert (dsD.sel(number=dsE.number).isel(step=0) == dsE.isel(step=-1)).all()

    ds = xr.concat([dsE[dsD.data_vars.keys()].isel(step=slice(None,-1)),
                    dsD.sel(number=dsE.number)], dim="step")
    ds = preprocess(ds, drop_wind_components=drop_wind_components,
                    temperature_in_C=temperature_in_C,
                    calculate_diffs=calculate_diffs, verbose=verbose)
    
    # check if the values in step xr.DataArray are unique
    steps = (ds.step / np.timedelta64(1, 'D')).round(2)
    assert steps.size == np.unique(steps).size
    
    logging.info(f"Loading complete")
    return ds, dsD