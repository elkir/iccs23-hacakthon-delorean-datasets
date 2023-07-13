# Describe: plot functions for the weather forecast
# import using the following command:
# from src.plot import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from ..preprocessing.reducers import average_over_shape

# color palette
# =========================================
colors_w10 = ["#9ecae1", "#6baed6", "#4292c6", "#2171b5", "#084594", "#08306b", ]
colors_w100 = ["#5d8aa8", "#135c45", "#1c7958", "#269791", "#34a5b3", "#43ccd8"]
colors_ssrd = ["#fdae6b", "#fd8d3c", "#f16913", "#d94801", "#8c2d04", "#a63603", ]
colors_t2m = [ "#221150", "#5f187f", "#982d80", "#d3436e", "#f8765c", "#febb81", ] # from the magma color palette

colors_ens = dict(w10=colors_w10, w100=colors_w100, ssrd=colors_ssrd, t2m=colors_t2m)


def get_country_record(country_name):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.shapereader as shpreader

    shpfilename = shpreader.natural_earth(resolution='110m',
                                        category='cultural',
                                        name='admin_0_countries')
    reader = shpreader.Reader(shpfilename)
    countries = reader.records()
    country = [country for country in countries if country.attributes['NAME_LONG'] == country_name][0]
    return country

### Plot the ensemble data line plots
## add types to the function


def plot_ens_tripleplot(ds, var, dsD=None, ax=None, load_full_D=False,
                      plots=["full_area", "country", "point"],
                      shape=None, country_name="United Kingdom",
                      latlon=(51.5, -0.1), point_name="London",
                      daily_average=False,
                      colors=None, **kwargs):
    """
    Plot the ensemble data line plots
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the ensemble data
    var : str
        Variable to plot
    dsD : xarray.Dataset, optional
        Dataset containing the deterministic data, by default None
    ax : matplotlib.axes.Axes, optional
        Axis to plot on, by default None
    load_full_D : bool, optional
        Whether to load the full D data, by default False
    plots : list, optional
        List of plots to make, by default ["full_area", "country", "point"]
    shape : shapely.geometry, optional
        Shape to average over, by default None
    country_name : str, optional
        Name of the country to plot, by default "United Kingdom"
    latlon : tuple, optional
        Latitude and longitude of the point to plot, by default (51.5, -0.1)
    point_name : str , optional
        Name of the point to plot, by default "London"
    daily_average : bool, optional
        Whether to average over the day, by default False
    colors : list, optional
        List of colors to use, by default None
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure of the plot
    ax : matplotlib.axes.Axes
        Axis of the plot
    """
    # =========================================
    ## Plot the ensemble data line plots
    # check if var is in the list of variables
    if var not in ds.data_vars:
        raise ValueError(f"Variable {var} not in dataset")
    
    num_ax = len(plots)

    if ax is None:
        fig,ax = plt.subplots(num_ax,1, figsize=(14, 8.5), sharex=True)
    elif isinstance(ax, mpl.axes.Axes):
        fig = ax.get_figure()
        ax = [ax]
    else: # assume it is a list of axes
        fig = ax[0].get_figure()
        assert len(ax) == num_ax, "Number of axes given does not match number of plots"
    
    if isinstance(country_name, list):
        countries = [get_country_record(country_name) for country_name in country_name]
    # if isinstance
        
    if "country" in plots:
        country = get_country_record(country_name)

    for i, axes,plot in zip(range(num_ax), ax, plots):
        # for last axes
        show_xlabels = i == num_ax-1
        shared_kwargs= dict(dsD=dsD, ax=axes,show_xlabels=show_xlabels, print_var_label=False,daily_average=daily_average, **kwargs)
        if plot == "full_area":
            plot_ens_lineplot(ds, var, 'full_area', **shared_kwargs)
        elif plot == "country":
            plot_ens_lineplot(ds, var, 'country', shape=country, country_name=country_name, **shared_kwargs) 
        elif plot == "point":
            plot_ens_lineplot(ds, var, 'point', latlon=latlon, point_name=point_name, **shared_kwargs)
        else:
            raise ValueError(f"Plot {plot} not recognized")

    # put the two plots on top of each other
    fig.subplots_adjust(hspace=0)
                
    # put title for the whole figure right on top
    # "ENS Extended: Wind speed (full resolution)"
    # close to the plots
    fig.suptitle(f"ENS Extended: {ds[var].attrs['long_name']} ({ds[var].attrs['units']})", fontsize=16, y=0.91)
    if ax is None:
        return fig, ax
    else:
        return ax


# a single axis line plot
def plot_ens_lineplot(ds,var,type, dsD=None, ax=None,
                      shape=None, country_name=None,
                      latlon=None,point_name=None,
                      daily_average=False,
                      colors=None, 
                      show_xlabels=False, print_var_label=True,
                      **kwargs):
    if ax is None:
        fig,ax = plt.subplots(1,1, figsize=(14, 4), sharex=True)
        show_xlabels = True
    
    # set the colors
    if var in colors_ens and colors is None:
        colors = colors_ens[var]
    elif colors is not None:
        colors = colors
    else:
        colors = plt.cm.tab10.colors
    ax.set_prop_cycle(mpl.cycler(color=colors))

    # rewrite the above as match statement
    match type:
        case "full_area":
            pass
        case "country":
            assert shape is not None, "Shapefile not given"
            assert country_name is not None, "Country name not given"
        case "point":
            assert latlon is not None, "Lat-lon not given"
        case _: # default
            raise ValueError(f"Type {type} not recognized")
    
    # check if var is in the list of variables
    if var not in ds.data_vars:
        raise ValueError(f"Variable {var} not in dataset")
    

    match type:
        case "full_area":
            if dsD is not None:
                dsD_plot= dsD[var].mean(dim=["latitude", "longitude"])
            ds_plot= ds[var].mean(dim=["latitude", "longitude"])
            label_location = f"over full area"
        case "country":
            if dsD is not None:
                dsD_plot= average_over_shape(dsD[var],shape.geometry)
            ds_plot= average_over_shape(ds[var],shape.geometry)
            label_location = f"over {country_name}"
        case "point":
            if dsD is not None:
                dsD_plot= dsD[var].sel(latitude=latlon[0], longitude=latlon[1] , method="nearest")
            ds_plot= ds[var].sel(latitude=latlon[0],longitude=latlon[1] , method="nearest")
            label_location = f"in {point_name}" if point_name is not None else f"at ({latlon[0]:.2f}, {latlon[1]:.2f})"
        
    if daily_average:
        if dsD is not None:
            dsD_plot = dsD_plot.resample(step="1D").mean()
            dsD_plot = dsD_plot.assign_coords(valid_time=dsD_plot.step+dsD_plot.time)
        ds_plot = ds_plot.resample(step="1D").mean()
        ds_plot = ds_plot.assign_coords(valid_time=ds_plot.step+ds_plot.time)
    drawstyle = "steps-post" if daily_average else "default"
    label_daily_average = "daily average" if daily_average else ""

    # Label for y axis
    if print_var_label:
        ylabel = f"{ds[var].attrs['long_name']} [{ds[var].attrs['units']}]\n{label_daily_average} {label_location}" 
    else:
        ylabel = f"{label_daily_average} {label_location} [{ds[var].attrs['units']}]"
        
    # plot the data
    if dsD is not None:
        dsD_plot.plot.line(x="valid_time", hue="number", ax=ax, linewidth=0.5, alpha=0.3, color="grey", drawstyle=drawstyle)
    ds_plot.plot.line(x="valid_time", hue="number", ax=ax, linewidth=1,drawstyle=drawstyle)
    
    # set ylim to the mid 90% of the data + 5% margin #TODO is this a good idea?
    ylim = dsD_plot.quantile([0.02, 0.98]).values if dsD is not None else ds_plot.quantile([0.02, 0.98]).values
    ylim = ylim[0] - 0.2*(ylim[1]-ylim[0]), ylim[1] + 0.2*(ylim[1]-ylim[0])
    ax.set_ylim(ylim)
    
    # plot dashed grey line at 0 (in the background) if 0 is in the mid 90% of the current ylim
    ylim = ax.get_ylim()
    if 0 > np.percentile(ylim, 5) and 0 < np.percentile(ylim, 95):
        ax.axhline(0, color="grey", linestyle="--", linewidth=1, zorder=-1)
    
    # axes details
    ax.set_ylabel(ylabel)
    ax.set_title("")
    # the legend is not needed
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ax.set_xlim([ds.valid_time.min(), ds.valid_time.max()])
    if show_xlabels:
        ax.set_xlabel("Date")
        # convert valid_time to a Pandas Series
        valid_time_pd = ds.valid_time.to_pandas()
        # calculate the weekly ticks for the x-axis
        xticks = valid_time_pd.min() + pd.to_timedelta(np.arange(0, (valid_time_pd.max() - valid_time_pd.min()).days, 7), unit='D')
        # set the calculated x-ticks on the x-axis
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks.strftime("%Y-%m-%d"))
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
    else:
        ax.set_xlabel("")
        ax.set_xticklabels([])

    if ax is None:
        return fig,ax
    else:
        return ax

