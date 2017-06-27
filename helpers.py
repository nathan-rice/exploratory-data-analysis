import numpy as np
import pandas as pd
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import HoverTool, LinearColorMapper, ColorBar, FuncTickFormatter, FixedTicker, AdaptiveTicker
from itertools import combinations, product
from scipy.stats import skew, kurtosis


def scatter_with_hover(df, x, y,
                       fig=None, cols=None, name=None, marker='x',
                       fig_width=500, fig_height=500, **kwargs):
    """
    Plots an interactive scatter plot of `x` vs `y` using bokeh, with automatic
    tooltips showing columns from `df`.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to be plotted
    x : str
        Name of the column to use for the x-axis values
    y : str
        Name of the column to use for the y-axis values
    fig : bokeh.plotting.Figure, optional
        Figure on which to plot (if not given then a new figure will be created)
    cols : list of str
        Columns to show in the hover tooltip (default is to show all)
    name : str
        Bokeh series name to give to the scattered data
    marker : str
        Name of marker to use for scatter plot
    **kwargs
        Any further arguments to be passed to fig.scatter

    Returns
    -------
    bokeh.plotting.Figure
        Figure (the same as given, or the newly created figure)

    Example
    -------
    fig = scatter_with_hover(df, 'A', 'B')
    show(fig)

    fig = scatter_with_hover(df, 'A', 'B', cols=['C', 'D', 'E'], marker='x', color='red')
    show(fig)

    Author
    ------
    Robin Wilson <robin@rtwilson.com>
    with thanks to Max Albert for original code example
    """

    # If we haven't been given a Figure obj then create it with default
    # size etc.
    if fig is None:
        fig = figure(width=fig_width, height=fig_height, tools=['box_zoom', 'reset', 'save'])

    # We're getting data from the given dataframe
    source = ColumnDataSource(data=df)

    # We need a name so that we can restrict hover tools to just this
    # particular 'series' on the plot. You can specify it (in case it
    # needs to be something specific for other reasons), otherwise
    # we just use 'main'
    if name is None:
        name = 'main'

    # Actually do the scatter plot - the easy bit
    # (other keyword arguments will be passed to this function)
    fig.scatter(x=x, y=y, source=source, name=name, marker=marker, **kwargs)

    # Now we create the hover tool, and make sure it is only active with
    # the series we plotted in the previous line
    hover = HoverTool(names=[name])

    if cols is None:
        # Display *all* columns in the tooltips
        hover.tooltips = [(c, '@' + c) for c in df.columns]
    else:
        # Display just the given columns in the tooltips
        hover.tooltips = [(c, '@' + c) for c in cols]

    # Finally add/enable the tool
    fig.add_tools(hover)

    return fig


def block_heatmap(df, height=600, width=900):
    """
    Generates a



    :param df:
        The Pandas DataFrame to render in block-heatmap style.
    :return:
        A Bokeh block heatmap figure modeled after example code.  The figure has additional properties, df for
        the plot data, and rect for the plot object.
    """
    # this colormap blatantly copied from the New York Times.
    colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
    mapper = LinearColorMapper(palette=colors, low=0, high=1)
    cols = {i: c for (i, c) in enumerate(df.columns)}
    index = {i: r for (i, r) in enumerate(df.index)}
    cols_by_rows = product(enumerate(df.columns), enumerate(df.index))
    data = np.array([[x, y, c, r, df.loc[r, c]] for ((x, c), (y, r)) in cols_by_rows])
    combination_df = pd.DataFrame(data, columns=["gene_id", "sample_id", "gene", "sample", "value"])
    source = ColumnDataSource(combination_df)

    fig = figure(title="Clustered Heatmap", toolbar_location="below", x_range=(0, len(df.columns)),
                 y_range=(0, len(df.index)), tools=["box_zoom", "pan", "reset", "save"], name="heatmap",
                 x_axis_location="above", plot_width=width, plot_height=height, active_drag="box_zoom")
    fig.rect(x="gene_id", y="sample_id", source=source, width=1, height=1,
                    fill_color={'field': 'value', 'transform': mapper}, line_color=None)

    fig.grid.grid_line_color = None
    fig.axis.axis_line_color = None
    fig.axis.major_tick_line_color = None
    fig.axis.major_label_text_font_size = "7pt"
    fig.axis.major_label_standoff = 0
    fig.xaxis.major_label_orientation = np.pi / 3

    fig.yaxis.formatter = FuncTickFormatter(code="""
        var labels = %s;
        return labels[tick] || '';
    """ % index)

    fig.xaxis.formatter = FuncTickFormatter(code="""
        var labels = %s;
        return labels[tick] || '';
    """ % cols)

    fig.yaxis.ticker = FixedTicker(ticks=list(index.keys()))
    fig.xaxis.ticker = AdaptiveTicker(mantissas=list(range(10)), min_interval=1, max_interval=5)

    hover = HoverTool(names=["heatmap"])
    hover.tooltips = [
        ('gene', '@gene'),
        ('sample', '@sample'),
        ('percentile', '@value%')
    ]
    fig.add_tools(hover)

    return fig


def add_dummy(dataframe, column_name):
    dummies = pd.get_dummies(dataframe[column_name], prefix="dummy_" + column_name)
    return pd.concat([dataframe, dummies], axis=1)


def filtered_combinations(columns, include_dummies=True, combine_dummies=False):
    def filter_if_dummies(t):
        a, b = t
        a_dummy = a.startswith("dummy_")
        b_dummy = b.startswith("dummy_")
        if not include_dummies and (a_dummy or b_dummy):
            return False
        if a_dummy and b_dummy:
            if combine_dummies:
                a_split = a.split("_")
                b_split = b.split("_")
                if not a_split[1] == b_split[1]:
                    return True
            return False
        return True

    return filter(filter_if_dummies, combinations(columns))


def generate_moment_statistics(data):
    data_skew = skew(data)
    data_kurtosis = kurtosis(data)
