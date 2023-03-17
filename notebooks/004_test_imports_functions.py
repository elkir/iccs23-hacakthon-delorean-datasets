# import the local module in src so that Pylance can find it
# # in particular the colors_ens dictionary
# 
# use PEP 328 import system
# grab the module in folder ../src
#%%%


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

dir_data = Path('../ecmwf-ens')
fn_E = dir_data /"mars_v04e_2017-01-02_Mon.grib"
fn_D = dir_data /"mars_v04d_2017-01-02_Mon.grib"

dir_fig = Path('../report/figures')
# =======================================
#%% Load the data
## Load the data
# =========================================

ds, dsD = load_ens_data_ED(fn_E, fn_D,
                           load_full_D=load_full_D,
                           drop_wind_components=drop_wind_components,
                           )

#%% print variables
## print variables
# =========================================
for v in ds.data_vars:
    # table format
    print(f"{v:6s} {ds[v].attrs['units']:10s} {ds[v].attrs['long_name']:30s}")
    


# %%
germany = get_country_record('Norway')

latlon_lisbon = (38.722252, -9.139337)
latlon_trondheim = (63.4305, 10.3951)
latlon_vienna = (48.20849, 16.37208)

colors = ["#FFC0CB", "#FFB6C1", "#FF69B4", "#FF1493", "#C71585"]
plot_ens_lineplot(ds, 'stl4', 'full_area')
# %%
plot_ens_lineplot(ds, 'w100', 'point',dsD=dsD , latlon=latlon_vienna, point_name='Vienna',print_var_label=False)

plot_ens_lineplot(ds, 'w100', 'full_area',dsD=dsD, latlon=latlon_vienna, point_name='Vienna',print_var_label=False)
# %%
fig, ax = plt.subplots(len(ds.data_vars), 1, sharex=True, figsize=(14, 16), dpi=400)

for i,var in enumerate(ds.data_vars):
    show_xlabels = i==len(ds.data_vars)-1
    plot_ens_lineplot(ds, var, 'full_area', ax=ax[i],print_var_label=True, dsD=dsD, show_xlabels=show_xlabels)
    ax[i].set_ylabel(f"{var}\n[{ds[var].attrs['units']}]")

# main title and small subtitle
# main: ENS Ensemble data, all variables, averaged over full area
# sub: (before converting radiation and runoff to Δ values)
fig.suptitle("ENS Ensemble data, all variables, averaged over full area", fontsize=16)
fig.text(0.5, 0.95, "(before converting radiation and runoff to Δ values)", ha='center', fontsize=10)

# the title and sub title are too high even with fig.subplots_adjust(top=0.9)
fig.subplots_adjust(top=0.85)
# no space between axes
fig.subplots_adjust(hspace=0)
# %%
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(14, 8), dpi=400)
plot_ens_lineplot(ds, 'w10', 'point',dsD=dsD , latlon=latlon_vienna, point_name='Vienna',print_var_label=False, ax=ax[0], show_xlabels=False)
plot_ens_lineplot(ds, 'w100', 'point',dsD=dsD , latlon=latlon_vienna, point_name='Vienna',print_var_label=False, ax=ax[1], show_xlabels=True)
fig.subplots_adjust(hspace=0)
fig.tight_layout()
# %%
ds.valid_time.max()
# %%
plot_ens_tripleplot(ds, 'w100',dsD=dsD )
plt.savefig(dir_fig / "004_w100_tripleplot_london.png")
# %%
plot_ens_tripleplot(ds, 't2m',dsD=dsD,country_name="Austria" , latlon=latlon_vienna, point_name='Vienna')

# %%
