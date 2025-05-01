import plotly.express as px
import pandas as pd
from typing import Union


def histogram(
    df: pd.DataFrame,
    column: str,
    title: str,
    nbins: int = 20
) -> px.histogram:

    fig = px.histogram(
        df,
        x=column,
        nbins=nbins,
        title=title,
        marginal="box"  
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
    
    fig = px.sunburst(
        df,
        path=path,
        values=values,
        title=title
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig
