import plotly.express as px
import pandas as pd
from typing import Union


def histogram(
    df: pd.DataFrame,
    column: str,
    title: str,
    nbins: int = 20
) -> px.histogram:
    """
    Create a Plotly histogram of `column` with `nbins` bins.
    """
    fig = px.histogram(
        df,
        x=column,
        nbins=nbins,
        title=title,
        marginal="box"  # show a boxplot on the side
    )
    fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
    return fig


def time_series_line(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    title: str,
    freq: str = None
) -> px.line:
    """
    Plot a time series line. If `freq` is provided (e.g. 'W' or 'M'),
    resample the data first by summing `value_col`.
    """
    ts = df.copy()
    ts[date_col] = pd.to_datetime(ts[date_col])
    ts = ts.set_index(date_col)
    if freq:
        ts = ts[value_col].resample(freq).sum().reset_index()
    else:
        ts = ts.reset_index()[[date_col, value_col]]

    fig = px.line(
        ts,
        x=date_col,
        y=value_col,
        title=title,
        markers=True
    )
    fig.update_layout(margin=dict(l=40, r=40, t=50, b=40))
    return fig


def bar_categories(
    df: pd.DataFrame,
    category_col: str,
    value_col: str = None,
    top_n: int = 10,
    title: str = None
) -> px.bar:
    """
    Plot counts (or aggregated `value_col`) for top N categories in `category_col`.
    If `value_col` is None, counts are used.
    """
    if value_col:
        agg = df.groupby(category_col)[value_col].sum()
    else:
        agg = df[category_col].value_counts()
    top = agg.sort_values(ascending=False).head(top_n).reset_index()
    top.columns = [category_col, value_col or "count"]
    fig = px.bar(
        top,
        x=category_col,
        y=value_col or "count",
        title=title or f"Top {top_n} by {category_col}",
        text=value_col or "count"
    )
    fig.update_layout(
        xaxis_title=category_col,
        yaxis_title=value_col or "count",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def sunburst_hierarchy(
    df: pd.DataFrame,
    path: list[str],
    values: Union[str, None] = None,
    title: str = None
) -> px.sunburst:
    """
    Sunburst chart for hierarchical data: `path` is a list of column names
    defining the hierarchy (e.g. ["genre", "month"]).
    `values` can be a column to sum, or omitted for counts.
    """
    fig = px.sunburst(
        df,
        path=path,
        values=values,
        title=title
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig
