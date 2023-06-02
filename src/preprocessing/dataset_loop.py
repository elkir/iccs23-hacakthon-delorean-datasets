

#%%  Import all the packages needed to explore grib data
import re
import xarray as xr
import numpy as np
import argparse
import gc
import logging
from pathlib  import Path
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# insert path to src folder no matter from where the file/notebook is run
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data_loading.load_ens import (load_ens_data_ED)
from src.preprocessing.reducers import calculate_variance, calculate_climatological_spatial_mean



# =======================================
#%% Flags and directories
## Flags and directories
# =========================================
load_full_D = True
drop_wind_components = True
calculate_diffs = True


#%%

def reducer_wrapper(filenameE, reducer, start_time=None):
    # check if the file is the e version: 'v(\d\d)e'
    assert re.search(r'v(\d\d)e', filenameE), "filenameE must be the E version"
    fn2_E = filenameE
    fn2_D = re.sub(r'v(\d\d)e', r'v\1d', filenameE)
    ds, dsD = load_ens_data_ED(fn2_E, fn2_D,
                            load_full_D=True,
                            drop_wind_components=True,
                            temperature_in_C=True,
                            calculate_diffs=True
                            )
    ds_reduced = reducer(ds.isel(step=slice(None,269-121)), start_time=start_time)
    dsD_reduced = reducer(dsD, start_time=start_time)
    del ds, dsD
    gc.collect()
    return ds_reduced, dsD_reduced


# TODO Proces files and aveger overlapping time steps


def process_files(export_filename, grib_filesE, reducer, start_time=None):
    all_reduced_E = []
    all_reduced_D = []

    logging.info(f"Processing {len(grib_filesE)} GRIB files")
    for i,filenameE in enumerate(grib_filesE):
        logging.info(f"Processing file {i+1}/{len(grib_filesE)}: {filenameE}")
        reduced_E, reduced_D = reducer_wrapper(filenameE, reducer,start_time=start_time)
        all_reduced_E.append(reduced_E)
        all_reduced_D.append(reduced_D)
        
    combined_reduced_E = xr.concat(all_reduced_E, dim="time")
    combined_reduced_D = xr.concat(all_reduced_D, dim="time")
    
    logging.info(f"Saving combined datasets to {export_filename}_E.nc and {export_filename}_D.nc")
    combined_reduced_E.to_netcdf(f"{export_filename}_E.nc")
    combined_reduced_D.to_netcdf(f"{export_filename}_D.nc")
    logging.info("Processing complete")

def main():
    parser = argparse.ArgumentParser(description='Process GRIB files and compute variance')
    parser.add_argument('export_filename', type=str, help='Base name for the output NetCDF files')
    parser.add_argument('grib_filesE', nargs='+', type=str, help='GRIB files (E) to process')
    args = parser.parse_args()

    export_filename = args.export_filename
    grib_filesE = args.grib_filesE

    # process_files(export_filename, grib_filesE, calculate_variance)
    process_files(export_filename, grib_filesE, calculate_climatological_spatial_mean)
    
    

if __name__ == "__main__":
    main()


# %%
