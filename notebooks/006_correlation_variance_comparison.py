

#%% Import all the packages needed to explore grib data
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy

from pathlib  import Path

import sys
# insert path to src folder no matter from where the notebook is run
sys.path.insert(0, "..")

# import my stuff:
from src.plotting.plot_ens import (colors_ens, plot_ens_lineplot,get_country_record,
                                   plot_ens_tripleplot)
from src.data_loading.load_ens import (calculate_wind_speed,
                                       load_ens_data_ED, average_over_shape) 
# auto reload imports
%load_ext autoreload
%autoreload 2
# wider plots
plt.rcParams['figure.figsize'] = [10, 5]
from IPython.core.interactiveshell import InteractiveShell
# just the last output
InteractiveShell.ast_node_interactivity = "last_expr"

#%% Flags and directories
## Flags and directories
# =========================================
load_full_D = True
drop_wind_components = True
validate_function = False
if validate_function:
    from src.tests.test_functions import validate_function_average_over_shape

dir_data = Path('../data')
fn_E = dir_data / "ecmwf-ens"/ "mars_v04e_2017-01-02_Mon.grib"
fn_D = dir_data / "ecmwf-ens"/ "mars_v04d_2017-01-02_Mon.grib"

fn_era5 = {
    2017: dir_data / "era5/ph4_yearly" / "europe-2017-era5.nc",
    2016: dir_data / "era5/ph4_yearly"/ "europe-2016-era5.nc",
}
dir_fig = Path('../report/figures')

#%% Load the data
## Load the data
# =========================================

ds, dsD = load_ens_data_ED(fn_E, fn_D,
                           load_full_D=load_full_D,
                           drop_wind_components=drop_wind_components,
                           )
#%%
ds_era5_2017 = xr.open_dataset(fn_era5[2017])
ds_era5_2016 = xr.open_dataset(fn_era5[2016])

#%%

#%% 
### 
# Calculate correlation betweem ensembles 1 and 2 (number field) in ds.t2m
# xarray create a new coordinate in the dataset called "numbers" with the value "1-2"

def get_correlation_time_series(ds, var, n_ens=5):
    corr_ds = xr.Dataset()


    for i in range(1,n_ens+1):
        for j in range(i+1,n_ens+1):
            # fill the dataset with the correlation between ensemble i and j
            # add each new data to relevant combination of i and j
            # pick the variable var from ds
            #f string formatting for two places: 01,...,51
            label = f"{i}-{j}" if n_ens < 10 else f"{i:02}-{j:02}"
            corr_ds[label] = xr.corr(ds[var].sel(number=i), ds[var].sel(number=j), dim=["latitude", "longitude"])
            
    # calculate the mean across all variables in ts_corr
    corr_ds = corr_ds.to_array(dim='combination')
    return corr_ds.mean(dim='combination'), corr_ds.std(dim='combination')
#%% Plot variance of all variables
## Plot variance of all variables
# =========================================
#line plot of all variables in the dataset in the same plot
# fig,ax = plt.subplots(1,1, figsize=(15, 10))

# calculate variance for each time step across all ensembles
variance = ds.var(dim="number").mean(dim=["latitude", "longitude"])
# varianceD = dsD.var(dim="number").mean(dim=["latitude", "longitude"])

z_fun = lambda x, axis: np.sqrt(np.var(x, axis=axis))/np.mean(x, axis=axis) #TODO: wrong function! mean is zero or negative sometimes
# apply z_fun over the dimension "number"

zval = xr.apply_ufunc(z_fun,ds, input_core_dims=[["number"]], kwargs={"axis":-1})
zval.mean(dim=["latitude", "longitude"]).to_array(dim='variable').plot.line(x="step", hue="variable", linewidth=1, ax=ax)
#%%
# if dsD is not None:
#     dsD_plot = varianceD.resample(step="1D").mean()
#     dsD_plot = dsD_plot.assign_coords(valid_time=dsD_plot.step+dsD_plot.time)
# ds_plot = variance.resample(step="1D").mean()
# ds_plot = ds_plot.assign_coords(valid_time=ds_plot.step+ds_plot.time)
# # z_value = (xr.apply_ufunc(z_fun,ds)).mean(dim=["latitude", "longitude"])

# ds_plot.to_array(dim='variable').plot.line(x="step", hue="variable", linewidth=1, ax=ax)
# dsD_plot.to_array(dim='variable').plot.line(x="step", hue="variable", linewidth=1, ax=ax, color="black", alpha=0.3)
# # yaxis range from 0 to 0.05
ax.set_ylim(0,1)
# # plot the mean and std as error bounds
# corr_vars.to_array(dim='variable').plot.line(x="valid_time", hue="variable", linewidth=1, ax=ax)
# # plot the std as error bounds around the mean
# # match the color of the line to the color of the error bounds
# colors = [ax.lines[i].get_color() for i in range(len(corr_vars.data_vars))]
# for i, var in enumerate(corr_vars.data_vars):
#     ax.fill_between(corr_vars.valid_time, corr_vars[var] + corr_std_vars[var], corr_vars[var] - corr_std_vars[var], color=colors[i], alpha=0.5)
# # (corr_vars + corr_std_vars/2).to_array(dim='variable').plot.line(x="valid_time", hue="variable", linewidth=1, ax=ax, alpha=0.5)
# # (corr_vars - corr_std_vars/2).to_array(dim='variable').plot.line(x="valid_time", hue="variable", linewidth=1, ax=ax, alpha=0.5)
# # use long name from ds variables as legend
# plt.legend([ds[var].long_name for var in ds.data_vars])
# # x tick lables horizontal and centered
# ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
# # title graph
# ax.set_title("Average Correlation between ensembles (all variables)")
# # save figure
# fig.savefig(dir_fig/"006-04_correlation_between_ensembles_all_variables.png", dpi=300)


#%%

def get_diff(da):
        # do a diff on the xarray Dataset and add the first value to the beginning
        # to get the same shape as the original dataset
        ddiff = da.diff(dim="step")
        ddiff = xr.concat([da.isel(step=0), ddiff], dim="step")
        return ddiff

da = ds.ssrd.sel(number=1)
da_diff = get_diff(da)



#%%
# sum
(ds.ssrd.sel(number=1).diff(dim="step")<0).any(dim=["latitude", "longitude"])




# column plot
ds_diff = ds.ssrd.sel(number=1).diff(dim="step")
# convert to hours integer
ds_diff['step'] = (ds.step[1:]/np.timedelta64(1, 'h')).astype(int)

# (ds_diff<0).sum(dim=["latitude", "longitude"]).plot()
# label peaks bigger 
# histogram of non zero negative values
# ignore nan values
ds_diff_negative = ds_diff.where(ds_diff<0)
# ds_diff_negative.plot.hist(bins=100

#get 4 max values
steps_sorted = ds_diff_negative.sum(dim=["latitude", "longitude"]).argsort()
#%%

#%%
# plot histogram of diff values, ignore zero values
# zeros to nan
ds_diff.where(ds_diff<0).sum(dim=["latitude", "longitude"]).plot()

# select random point in france
ds.sel(latitude=48.8566, longitude=2.3522, method="nearest").plot.line(x="step", hue="number", linewidth=1, ax=ax)

#%%
fig,axis = plt.subplots(3,2,figsize=(10,15) , sharex=True, sharey=True, subplot_kw={'projection': ccrs.PlateCarree()})
def plot_negative_map(ds, ax):
    # include subtle topography
    ds.plot.contourf(ax=ax, levels=20, add_colorbar=False, cmap="Blues")
    # plot the negative values
    ds.where(ds<0).plot.contourf(ax=ax, levels=20, add_colorbar=False, cmap="Reds")
    ax.coastlines()
# plot the first 4 steps of steps_sorted
for i,ax in enumerate(axis.flatten()):
    plot_negative_map(ds_diff_negative.isel(step=steps_sorted[i]), ax=ax)
    ax.set_title(f"Step {steps_sorted[i].values}")
    

#%%
# create a video of all the steps in ds_diff
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from matplotlib import colors
def animate(ds):
    fig,ax = plt.subplots(1,1, figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    # set the bounds of the colorbar
    vmin = ds.min().values
    vmax = ds.max().values*08
    divnorm=colors.TwoSlopeNorm(vmin=vmin/2, vcenter=0, vmax=vmax)
    # replace 0 with white
    
    def animate_frame(i):
        ax.clear()
        # colormap diverges at 0
    
        ds.isel(step=i).plot.contourf(ax=ax, levels=20, add_colorbar=False, cmap="RdBu_r", norm=divnorm)    
        ax.set_title(f"{ds.valid_time[i].values}")
        ax.coastlines()
    variable_frame_length = ds.step.diff(dim="step").values
    
    anim = FuncAnimation(fig, animate_frame, frames=ds.step.size)
    return anim

animate(ds_diff)

#%% =====================================
### Get and Plot correlations for all variables in ds 
# =======================================
### Calculate:
# repeat the above but pefrom the correlation across all variables in ds 
# keep the step as a dimension
corr_vars = xr.Dataset()
corr_std_vars = xr.Dataset()

# for all variables in ds extract the correlation time series
for var in ds.data_vars:
    corr_vars[var],corr_std_vars[var] = get_correlation_time_series(ds, var)
    corr_vars_D[var],corr_std_vars_D[var] = get_correlation_time_series(dsD, var, n_ens=50)
#%% 
## Plot:
#line plot of all variables in the dataset in the same plot
fig,ax = plt.subplots(1,1, figsize=(15, 10))

# plot the mean and std as error bounds
corr_vars.to_array(dim='variable').plot.line(x="valid_time", hue="variable", linewidth=1, ax=ax)
# plot the std as error bounds around the mean
# match the color of the line to the color of the error bounds
colors = [ax.lines[i].get_color() for i in range(len(corr_vars.data_vars))]
for i, var in enumerate(corr_vars.data_vars):
    ax.fill_between(corr_vars.valid_time, corr_vars[var] + corr_std_vars[var], corr_vars[var] - corr_std_vars[var], color=colors[i], alpha=0.5)
# (corr_vars + corr_std_vars/2).to_array(dim='variable').plot.line(x="valid_time", hue="variable", linewidth=1, ax=ax, alpha=0.5)
# (corr_vars - corr_std_vars/2).to_array(dim='variable').plot.line(x="valid_time", hue="variable", linewidth=1, ax=ax, alpha=0.5)
# use long name from ds variables as legend
plt.legend([ds[var].long_name for var in ds.data_vars])
# x tick lables horizontal and centered
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
# title graph
ax.set_title("Average Correlation between ensembles (all variables)")
# save figure
fig.savefig(dir_fig/"006-04_correlation_between_ensembles_all_variables.png", dpi=300)

