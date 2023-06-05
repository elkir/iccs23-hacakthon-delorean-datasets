
# vizual test for the functions
def validate_function_average_over_shape(ds, shape_uk,var="t2m", number=1):
    # compare direct plot of london and an average over a small box around london and the total average
    # use ds t2m and ensemble 1
    import shapely.geometry
    import matplotlib.pyplot as plt
    from src.preprocessing.reducers import  average_over_shape 
    box_around_london = shapely.geometry.box(-0.5, 51, 0.5, 52)
    if var=="t2m":
        (ds.t2m.sel(number=1, longitude=-0.1, latitude=51.5, method="nearest") - 273.15).plot.line(x="valid_time", label="London")
        (average_over_shape(ds.t2m.sel(number=1), box_around_london) - 273.15).plot.line(x="valid_time", label="Box around London")
        (average_over_shape(ds.t2m.sel(number=1), shape_uk) - 273.15).plot.line(x="valid_time", label="UK")
        (ds.t2m.sel(number=1).mean(dim=["latitude", "longitude"]) - 273.15).plot.line(x="valid_time", label="Europe")
    elif var=="w10":
        (ds.w10.sel(number=1, longitude=-0.1, latitude=51.5, method="nearest")).plot.line(x="valid_time", label="London")
        (average_over_shape(ds.w10.sel(number=1), box_around_london)).plot.line(x="valid_time", label="Box around London")
        (average_over_shape(ds.w10.sel(number=1), shape_uk)).plot.line(x="valid_time", label="UK")
        (ds.w10.sel(number=1).mean(dim=["latitude", "longitude"])).plot.line(x="valid_time", label="Europe")
    plt.legend()
    plt.show()