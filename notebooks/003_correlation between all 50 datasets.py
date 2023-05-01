### =======================================
"""
This notebook does: 
- determine the correlation of all 50 dataset
- looks at wind, temperature, humidity, and ssrd
- looks at 3 levels of detail: london, UK, and full extent of the european cutout
- (optional) looks at the correlation with respect to inter-annual variability
"""

### =======================================
#%%  Import all the packages needed to explore grib data
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy

from pathlib  import Path

import sys
sys.path.insert(0, "..")

# import my stuff:
from src.plotting.plot_ens import (colors_ens, plot_ens_lineplot,get_country_record)
from src.data_loading.load_ens import (calculate_wind_speed,
                                       load_ens_data_ED, average_over_shape) 

# auto reload imports
%load_ext autoreload
%autoreload 2


#%% wider plots
# wider plots
plt.rcParams['figure.figsize'] = [10, 5]
from IPython.core.interactiveshell import InteractiveShell
# just the last output
InteractiveShell.ast_node_interactivity = "last_expr"


# =======================================
#%% Flags and directories
## Flags and directories
# =========================================
load_full_D = True
drop_wind_components = True
validate_function = False
if validate_function:
    from src.tests.test_functions import validate_function_average_over_shape

dir_data = Path('../data/ecmwf-ens')
fn_E = dir_data /"mars_v04e_2017-01-02_Mon.grib"
fn_D = dir_data /"mars_v04d_2017-01-02_Mon.grib"

dir_fig = Path('../report/figures')
# =======================================
#%% Load the data
## Load the data
# =========================================

ds, dsD = load_ens_data_ED(fn_E, fn_D,
                           load_full_D=load_full_D,
                           drop_wind_components=drop_wind_components)

#%% print variables
## print variables
# =========================================
for v in ds.data_vars:
    # table format
    print(f"{v:6s} {ds[v].attrs['units']:10s} {ds[v].attrs['long_name']:30s}")

# =======================================
#%% function to take a xarray dataarray and average over the 2D space masked by a shape (shapely multipolygon)
## function to take a xarray dataarray and average over the 2D space masked by a shape (shapely multipolygon)
# =========================================
if validate_function:
    shape_uk = get_country_record("United Kingdom")
    validate_function_average_over_shape(ds, shape_uk, var="t2m", number=1)

# =======================================
#%% Two temperature plots (mean and London)
## Two temperature plots (mean and London)
## =======================================

fig,ax = plt.subplots(2,1, figsize=(18, 10), sharex=True)

if load_full_D:
# plot D data as thin transparent grey lines
    (dsD.t2m.mean(dim=["latitude", "longitude"]) - 273.15).plot.line(x="valid_time", hue="number", ax=ax[0], linewidth=0.5, alpha=0.3, color="grey")
    (dsD.t2m.sel(longitude=-0.1, latitude=51.5, method="nearest") - 273.15).plot.line(x="valid_time", hue="number", ax=ax[1], linewidth=0.5, alpha=0.3, color="grey")
# average the temperature across latitudes and longitudes (convert from K to C)
(ds.t2m.mean(dim=["latitude", "longitude"]) - 273.15).plot.line(x="valid_time", hue="number", ax=ax[0], linewidth=1)
# plot temperature for the ensembles as line plots for London
(ds.t2m.sel(longitude=-0.1, latitude=51.5, method="nearest") - 273.15).plot.line(x="valid_time", hue="number", ax=ax[1], linewidth=1)


# put the "Mean temperature [C]" and "Temperature in London [C]" instead of the default y axis labels
ax[0].set_ylabel("Averaged temperature (full area) [C]")
ax[1].set_ylabel("Temperature in London [C]")

#   for both plots
for a in ax:
    
    # clear titles
    a.set_title("")
    # legend off for both plots
    a.legend().set_visible(False)
    # set the x range to min and max of "valid time"
    a.set_xlim([ds.valid_time.min(), ds.valid_time.max()])

    # add 0 horizontal dashed line
    a.plot([a.get_xlim()[0], a.get_xlim()[1]], [0, 0], "k--", linewidth=2, alpha=0.5) 

# put the two plots on top of each other
fig.subplots_adjust(hspace=0)

# set the x axis label as "Date" and set tick labels horizontal and centered
ax[1].set_xlabel("Date")
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=0, ha="center")
    
# put title for the whole figure right on top
# "ENS Extended: Temperature (full resolution)"
# close to the plots
fig.add_artist(plt.figtext(0.5, 0.90, "ENS Extended: Temperature (full resolution)", ha="center", fontsize=20))

#overwrite only if load_full_D
# if load_full_D:
#     fig.savefig(dir_fig/"003-01_ens_extended_temperature.png", dpi=300)


# =======================================
#%% Plot the same as the above cell but for the 10m wind speed
### Plot the same as above but for the 10m wind speed
# =========================================
# Copy above cell and replace:
# "t2m" -> "w10"
# "Temperature" -> "Wind speed"
# and delete the -275.15 conversion
# context color cycle for the wind speed plot (blue green)
with mpl.rc_context({"axes.prop_cycle": mpl.cycler(color=colors_blue_green)}):
    fig,ax = plt.subplots(3,1, figsize=(14, 9), sharex=True)

    # if load_full_D:
    #     # plot D data as thin transparent grey lines
    #     (dsD.w10.mean(dim=["latitude", "longitude"])).plot.line(x="valid_time", hue="number", ax=ax[0], linewidth=0.5, alpha=0.3, color="grey")
    #     (average_over_shape(dsD.w10,shape_uk)).plot.line(x="valid_time", hue="number", ax=ax[1], linewidth=0.5, alpha=0.3, color="grey")
    #     (dsD.w10.sel(longitude=-0.1, latitude=51.5, method="nearest")).plot.line(x="valid_time", hue="number", ax=ax[2], linewidth=0.5, alpha=0.3, color="grey")


    # temporarily modify the colour cycle to use the new colors
    ###plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors_blue_green)
    (ds.w10.mean(dim=["latitude", "longitude"])).plot.line(x="valid_time", hue="number", ax=ax[0], linewidth=1)
    ###plt.rcParams["axes.prop_cycle"] = plt.cycler(color=plt.cm.tab10.colors)
    (average_over_shape(ds.w10,shape_uk)).plot.line(x="valid_time", hue="number", ax=ax[1], linewidth=1)
    ###plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors_blue_green)
    (ds.w10.sel(longitude=-0.1, latitude=51.5, method="nearest")).plot.line(x="valid_time", hue="number", ax=ax[2], linewidth=1)

    ###plt.rcParams["axes.prop_cycle"] = plt.cycler(color=plt.cm.tab10.colors)

    #put the "Mean wind speed [m/s]" and "Wind speed in London [m/s]" instead of the default y axis labels
    ax[0].set_ylabel("Averaged (full area) [m/s]")
    ax[1].set_ylabel("Averaged (UK) [m/s]")
    ax[2].set_ylabel("In London [m/s]")

    #   for both plots

    for a in ax:
        # clear titles
        a.set_title("")
        # legend off for both plots
        a.legend().set_visible(False)
        # set the x range to min and max of "valid time"
        a.set_xlim([ds.valid_time.min(), ds.valid_time.max()])

    # put the two plots on top of each other
    fig.subplots_adjust(hspace=0)
    # set the x axis label as "Date" and set tick labels horizontal and centered
    ax[-1].set_xlabel("Date")
    ax[-1].set_xticklabels(ax[-1].get_xticklabels(), rotation=0, ha="center")
        
    # put title for the whole figure right on top
    # "ENS Extended: Wind speed (full resolution)"
    # close to the plots
    fig.add_artist(plt.figtext(0.5, 0.90, "ENS Extended: 10m wind speed", ha="center", fontsize=20))

fig,ax = plt.subplots(1,1, figsize=(14, 9), sharex=True)
ax = plot_t2m(ax)
ax.set_ylabel("Temperature in Zurich [C]")

# overwrite only if load_full_D
# if load_full_D:
#    fig.savefig(dir_fig/"003-02_ens_extended_wind_speed.png", dpi=300)
# reset color cycle
# plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab10.colors)
# =======================================
# %%
#plot 5 different sine waves with different colors
#compress the above code into a loop

def plot_sine_waves(j, ax):
    x = np.linspace(0, 2*np.pi, 100)
    for i in range(5):
        y = np.sin(x + i*np.pi/2)
        ax.plot(x,y)
    return ax

# plot from ds.t2m
# t2m is converted from Kelvin to C
# t2m is shown for a location: zurich
def plot_t2m(ax):
    (ds.t2m - 273.15).sel(longitude=8.55, latitude=47.37, method="nearest").plot.line(x="valid_time", hue="number", ax=ax, linewidth=1)
    ax.legend().set_visible(False)
    return ax


for j in range(6):
    colours = plt.cm.tab10.colors if j % 2 == 0 else colors_blue_green
    with mpl.rc_context({"axes.prop_cycle": plt.cycler(color=colours)}):
    
        fig,ax = plt.subplots(1,1, figsize=(14, 2))
        # ax = plot_sine_waves(j, ax)
        ax = plot_t2m(ax)
        title = str(j+1) + " - " + ("blue" if j % 2 == 0 else "colors")
        ax.set_title(title)
        


#%% Temperture plots as daily means
### Temperture plots as daily means
# =========================================
# shared x axis
# convert step datetime64[ns] to datetime64[D] and use as x axis

fig,ax = plt.subplots(2,1, figsize=(18, 10), sharex=True)

# calculate daily mean temperature for the whole area and for London
daily_mean_average = (ds.t2m.mean(dim=["latitude", "longitude"]) - 273.15).resample(step="1D").mean()
daily_mean_london = (ds.t2m.sel(longitude=-0.1, latitude=51.5, method="nearest") - 273.15).resample(step="1D").mean()
# get back valid_time from step and time 
daily_mean_average = daily_mean_average.assign_coords(valid_time=daily_mean_average.step + daily_mean_average.time)
daily_mean_london = daily_mean_london.assign_coords(valid_time=daily_mean_london.step + daily_mean_london.time)

if load_full_D:
    # calculate for D
    daily_mean_average_D = (dsD.t2m.mean(dim=["latitude", "longitude"]) - 273.15).resample(step="1D").mean()
    daily_mean_london_D = (dsD.t2m.sel(longitude=-0.1, latitude=51.5, method="nearest") - 273.15).resample(step="1D").mean()
    daily_mean_average_D = daily_mean_average_D.assign_coords(valid_time=daily_mean_average_D.step + daily_mean_average_D.time)
    daily_mean_london_D = daily_mean_london_D.assign_coords(valid_time=daily_mean_london_D.step + daily_mean_london_D.time)


# plot everything
if load_full_D:
    # plot as hlines
    daily_mean_average_D.plot.line(x="valid_time", hue="number", ax=ax[0], linewidth=0.5, alpha=0.3, color="grey", drawstyle="steps-mid")
    daily_mean_london_D.plot.line(x="valid_time", hue="number", ax=ax[1], linewidth=0.5, alpha=0.3, color="grey", drawstyle="steps-mid")
daily_mean_average.plot.line(x="valid_time", hue="number", ax=ax[0], linewidth=2, drawstyle="steps-mid")
daily_mean_london.plot.line(x="valid_time", hue="number", ax=ax[1], linewidth=2, drawstyle="steps-mid")
                

# put the "Mean temperature [C]" and "Temperature in London [C]" instead of the default y axis labels
ax[0].set_ylabel("Averaged temperature (full area) [C]")
ax[1].set_ylabel("Temperature in London [C]")

#   for both plots
for a in ax:
    # clear titles
    a.set_title("")
    # legend off for both plots
    a.legend().set_visible(False)
    

    # limit range to data
    a.set_xlim([daily_mean_average.valid_time.min(), daily_mean_average.valid_time.max()])
    
    # add 0 horizontal dashed line
    a.plot([a.get_xlim()[0], a.get_xlim()[1]], [0, 0], "k--", linewidth=2, alpha=0.5) 

# put the two plots on top of each other
fig.subplots_adjust(hspace=0)

# set the x axis label as "Date" and set tick labels horizontal and centered
ax[1].set_xlabel("Date")
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=0, ha="center")

# put title for the whole figure right on top
fig.add_artist(plt.figtext(0.5, 0.90, "ENS Extended: Daily mean temperature", ha="center", fontsize=20))
# save and show
#overwrite only if load_full_D
# if load_full_D:
#     fig.savefig(dir_fig/"003-02_ens_extended_daily_mean_temperature.png", dpi=300)


#%% ------------------------------------
# # describe variables

# loop for all variables, print long name, units and 
for var in ds.data_vars:
    #catch atribute error if GRIB_paramId is not present
    try:
        id = str(ds[var].GRIB_paramId)
    except AttributeError:
        id = "xxxxxx"
    #check if id is 6 digits
    if len(id) == 6:
        id= id[3:6]+"."+id[0:3]
    else :
        id = id[0:3] + ".128"
    print(f"{id} - {var:4} - {ds[var].long_name} [{ds[var].units}]")

# 

# =======================================
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


avg_t2m_corr, std_t2m_corr = get_correlation_time_series(ds, "t2m")
# do the same for 50 ensemble members of ds_D
avg_t2m_corr_D, std_t2m_corr_D = get_correlation_time_series(dsD, "t2m", n_ens=50)



# =====================================
#%% Plot the correlation between ensembles with error bounds
## Plot the correlation between ensembles with error bounds
# =====================================

# plot the mean and std as error bounds
fig, ax = plt.subplots(1,1, figsize=(15, 5))
avg_t2m_corr.plot.line(x="valid_time", ax=ax, linewidth=1)
# plot the std as error bounds around the mean
ax.fill_between(avg_t2m_corr.valid_time, avg_t2m_corr + std_t2m_corr, avg_t2m_corr - std_t2m_corr, alpha=0.5)

# plot the D data
ax.plot(avg_t2m_corr_D.valid_time, avg_t2m_corr_D, color="red", linewidth=1)
ax.fill_between(avg_t2m_corr_D.valid_time, avg_t2m_corr_D + std_t2m_corr_D, avg_t2m_corr_D - std_t2m_corr_D, alpha=0.5, color="red")


# x tick lables horizontal and centered
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")

# plot vertical grey dashed line before the first valid time of the D data
cutoff = avg_t2m_corr_D.valid_time[0].values
ax.axvline(cutoff, color="grey", linestyle="--", linewidth=2)
# add a small text right of this line "→ 50 ensemble members"
ax.text(cutoff + np.timedelta64(10, 'h'), 0.51, "→ 50 ensemble members", fontsize=12)

# set title as "Average Correlation between ensembles (t2m)"
ax.set_title("Average Correlation between ensembles (t2m)")
# fig.savefig(dir_fig/"003-03_correlation_between_ensembles_t2m.png", dpi=300)

# =====================================
#%% Get and Plot correlations for all variables in ds 
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
# plot vertical grey dashed line at two weeks after the start of the dataset
ax.axvline(x=ds.valid_time[0] + np.timedelta64(14, 'D'), color='grey', linestyle='--')
# title graph
ax.set_title("Average Correlation between ensembles (all variables)")


# save figure
# fig.savefig(dir_fig/"003-04_correlation_between_ensembles_all_variables.png", dpi=300)


# =====================================
#%% Plot correlations for all variables in ds (log scale)
### Plot correlations for all variables in ds (log scale)
## =======================================
#line plot of all variables in the dataset in the same plot

corr_vars_inv = 1- corr_vars

# drop geopotential
corr_vars_inv = corr_vars_inv.drop_vars("z")
corr_std_vars_inv = corr_std_vars.drop_vars("z")


fig,ax = plt.subplots(1,1, figsize=(15, 10))



### Plot the mean and std as error bounds
corr_vars_inv.to_array(dim='variable').plot.line(x="valid_time", hue="variable", linewidth=1, ax=ax)
# plot the std as error bounds around the mean
# match the color of the line to the color of the error bounds
colors = [ax.lines[i].get_color() for i in range(len(corr_vars_inv.data_vars))]
for i, var in enumerate(corr_vars_inv.data_vars):
    ax.fill_between(corr_vars_inv.valid_time, corr_vars_inv[var] + corr_std_vars_inv[var], corr_vars_inv[var] - corr_std_vars_inv[var], color=colors[i], alpha=0.5)

# use long name from ds variables as legend
plt.legend([ds[var].long_name for var in corr_vars_inv.data_vars])
# log y axis
plt.yscale("log")

## x tick lables horizontal and centered
## title graph
## save figure
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
ax.set_title("Correlation between ensembles (1-log(C) scale)")
# fig.savefig(dir_fig/"003-05_correlation_between_ensembles_all_variables_logscale.png", dpi=300)


# %% =====================================

#select random (!) sample of 16 spatial points
ts1 = ds.t2m.sel(number=1).isel(latitude=slice(0,100,10), longitude=slice(0,100,10))
ts2 = ds.t2m.sel(number=2).isel(latitude=slice(0,100,10), longitude=slice(0,100,10))
# # plot scatter plot of t2m ensemble 1 and 2
fig, ax = plt.subplots(1,1, figsize=(8,8))
# scatter plot but color the points by the valid time


cval = ts1.step.values/np.timedelta64(1, 'D')
# tile cval to match the shape of ts1 
cval = np.tile(cval, (ts1.shape[1], ts1.shape[2], 1)).transpose(2,0,1)
ax.scatter(ts1[::-1], ts2[::-1], s=1, c=cval,alpha=0.5)
# # plot diagonal line
ax.plot([ts1.min(), ts1.max()], [ts1.min(), ts1.max()], "--", color="gray", linewidth=1)
# add colorbar
cbar  = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(cval.min(), cval.max()), cmap="viridis_r"), ax=ax)
cbar.set_label("days")
# add labels
ax.set_xlabel("t2m ensemble 1")
ax.set_ylabel("t2m ensemble 2")
# # %%

# =====================================
#%% Plot scatter plot of all variables in ds
### Plot scatter plot of all variables in ds
# =======================================


# create a grid of subplots
fig, axes = plt.subplots(3,4, figsize=(20,15))
# flatten the axes array to a list
axes = axes.flatten()


# select random (!) sample of 16 spatial points
ts1 = ds.sel(number=1).isel(latitude=slice(0,100,10), longitude=slice(0,100,10))
ts2 = ds.sel(number=2).isel(latitude=slice(0,100,10), longitude=slice(0,100,10))
cval = ts1.step.values/np.timedelta64(1, 'D')
# tile cval to match the shape of ts1 random variable
 
cval = np.tile(cval, (ts1.t2m.shape[1], ts1.t2m.shape[2], 1)).transpose(2,0,1)

# loop over all variables in the dataset
for i, var in enumerate(ds.data_vars):
    s1 = ts1[var]
    s2 = ts2[var]
    
    # select the axis to plot to
    ax = axes[i]
    # scatter plot but color the points by the valid time
    
    #ax.scatter(s1, s2, s=1, c=cval,alpha=0.5)
    # scatter from end to start
    ax.scatter(s1[::-1], s2[::-1], s=1, c=cval,alpha=0.5, cmap="viridis")
    
    # plot diagonal line
    ax.plot([s1.min(), s1.max()], [s1.min(), s1.max()], "--", color="gray", linewidth=1)

    # no labels
    ax.set_xlabel("")
    ax.set_ylabel("")
    
    #square plot
    ax.set_aspect('equal', 'box')
    
    # add title
    ax.set_title(f"{ds[var].long_name} [{ds[var].units}]")

# remove 16th axes and move the last three plots to the right
fig.delaxes(axes[-1])
    
# add a shared colorbar (smaller size)
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
#invert the colorbar color order
cbar  = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(cval.min(), cval.max()), cmap="viridis_r"), cax=cbar_ax)
cbar.set_label("days")

# add a title for the whole figure
# "Scatter plot between ensemble members 1 and 2"
fig.suptitle(f"Scatter plot between ensemble members 1 and 2", fontsize=20)
# save figure
# fig.savefig(dir_fig/"003-06_scatter_ensemble_1-2.png", dpi=300)





# ====================
#%% Explore monotonically increasing variables
### Explore monotonically increasing variables
# =======================
# depends on the previous cell
# =======================
# find which variables are only increasing
diff_vars = ["ssrd","strd","ssr", "ro"]
# create a new ds_diff with the diff_vars variables only, copying the attributes, and taking the diff("step") function
ds_diff = ds[diff_vars].copy()
ds_diff.attrs = ds.attrs
ds_diff = ds_diff.diff("step")
# set all values that are not increasing to NaN
ds_diff = ds_diff.where(ds_diff > 0, np.nan)

ds_diff.where(ds_diff > 0, 0)

# =====================================
#%% plot a 2d map for ensembles 1 and 2 for the 200th time step, variable 10m wind speed
### plot a 2d map for ensembles 1 and 2 for the 200th time step, variable 10m wind speed



fig, axes = plt.subplots(1,2, figsize=(15,5), subplot_kw={"projection": ccrs.PlateCarree()})
for i, ens in enumerate([1,2]):
    # select the axis to plot to
    ax = axes[i]
    # plot histogram ignoring NaNs
    ds.isel(number=ens, step=200).w10.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap="viridis")
    # add title
    ax.set_title(f"ensemble {ens}")
    # add coastlines
    ax.coastlines()
    # add gridlines
    ax.gridlines()
# print the step date as a title
fig.suptitle(ds.valid_time[200].values)

fig, axes = plt.subplots(1,2, figsize=(15,5), subplot_kw={"projection": ccrs.PlateCarree()})
for i, ens in enumerate([1,2]):
    # select the axis to plot to
    ax = axes[i]
    # plot histogram ignoring NaNs
    ds.isel(number=ens, step=200).w100.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap="viridis")
    # add title
    ax.set_title(f"ensemble {ens}")
    # add coastlines
    ax.coastlines()
    # add gridlines
    ax.gridlines()
# print the step date as a title
fig.suptitle(ds.valid_time[200].values)

var = "ssrd"
fig, axes = plt.subplots(1,2, figsize=(15,5), subplot_kw={"projection": ccrs.PlateCarree()})
for i, ens in enumerate([1,2]):
    # select the axis to plot to
    ax = axes[i]
    # plot with colorbar from 0 to max
    ds_diff.isel(number=ens, step=202)[var].plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap="magma", vmin=0)
    # add title
    ax.set_title(f"ensemble {ens}")
    # add coastlines
    ax.coastlines()
    # add gridlines
    ax.gridlines()
# print the step date as a title
fig.suptitle(ds.valid_time[202].values)
# %%

# plot the 2d map for the mean of ssrd for all ensembles and all time steps
fig, ax = plt.subplots(figsize=(10,5), subplot_kw={"projection": ccrs.PlateCarree()})
# plot histogram ignoring NaNs
ds[var].isel(step=107).mean(dim=["number"]).plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap="magma")
# add coastlines
ax.coastlines()
# add gridlines
ax.gridlines()


# ====================
#%% plot histogram for each of the 11 variables in ds_diff in a 4x3 grid
### plot histogram for each of the 11 variables in ds_diff in a 4x3 grid

# ignore the lowest 10% and the highest 10% of the data
quantiles = ds_diff.quantile([0.1, 0.9], dim="step")
ds_plot = ds_diff.where((ds_diff > quantiles.isel(quantile=0)) & (ds_diff < quantiles.isel(quantile=1)), drop=True)

#%%
fig, axes = plt.subplots(3,4, figsize=(20,15))
# flatten the axes array to a list
axes = axes.flatten()
for i, var in enumerate(ds.data_vars):
    # select the axis to plot to
    ax = axes[i]
    # plot histogram ignoring NaNs
    ds_plot[var].plot.hist(ax=ax, bins=100, density=True)
    # add title
    ax.set_title(ds[var].long_name)
    # plot a grey vertical line at 0
    ax.axvline(0, color="gray", linewidth=1)
    # set y axis to log scale
    ax.set_yscale("log")


# ====================
#%% intermean average
### intermean average
# =======================

# find intermean average
ds_mean = ds.mean(dim="number")
ds_deviation = ds - ds_mean

# find correlation between ensemble members for ds_deviation

### Calculate:
# repeat the above but pefrom the correlation across all variables in ds 
# keep the step as a dimension
corr_vars_dev = xr.Dataset()
corr_std_vars_dev = xr.Dataset()

# for all variables in ds extract the correlation time series
for var in ds_deviation.data_vars:
    corr_vars_dev[var],corr_std_vars_dev[var] = get_correlation_time_series(ds_deviation, var)

#%% 
## Plot:
#line plot of all variables in the ataset in the same plot
fig,ax = plt.subplots(1,1, figsize=(15, 10))

d = corr_vars_dev
d_std = corr_std_vars_dev
# drop z variable
d = d.drop_vars("z")
d_std = d_std.drop_vars("z")

# plot the mean and std as error bounds
d.to_array(dim='variable').plot.line(x="valid_time", hue="variable", linewidth=1, ax=ax)
# plot the std as error bounds around the mean
# match the color of the line to the color of the error bounds
colors = [ax.lines[i].get_color() for i in range(len(d.data_vars))]
for i, var in enumerate(d.data_vars):
    ax.fill_between(d.valid_time, d[var] + d_std[var], d[var] - d_std[var], color=colors[i], alpha=0.5)
# (corr_vars + corr_std_vars/2).to_array(dim='variable').plot.line(x="valid_time", hue="variable", linewidth=1, ax=ax, alpha=0.5)
# (corr_vars - corr_std_vars/2).to_array(dim='variable').plot.line(x="valid_time", hue="variable", linewidth=1, ax=ax, alpha=0.5)
# use long name from ds variables as legend
plt.legend([ds[var].long_name for var in d.data_vars])
# x tick lables horizontal and centered
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
# title graph
ax.set_title("Average Correlation between Deviation of ensembles from mean")
# add an entry to to legend "Geopotential Height (ommited)"
ax.legend([ds[var].long_name for var in d.data_vars]+ ["Geopotential Height (omitted)"])

# save into the directory as everything else, same file naming convention
# fig.savefig(f"{dir}/003-07_correlation_after_subtracting_mean.png", dpi=300)

# %%
