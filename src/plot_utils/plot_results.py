import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px


def plot_grouped_results(
    df, names, type=None, title="", print_output=False, fig_size=(600, 600)
):
    if type:
        grouped = (
            df[df["name"].isin(names) & (df.type == type) & (df.concerned_year != 2024)]
            .groupby(["name", "concerned_year"])["all_mean"]
            .mean()
        )
    else:
        grouped = (
            df[df["name"].isin(names) & (df.concerned_year != 2024)]
            .groupby(["name", "concerned_year"])["all_mean"]
            .mean()
        )

    grouped_c = grouped.copy().reset_index()
    d = {}
    for x, y in grouped_c.groupby("name"):
        values = y[y["concerned_year"] == 2023]["all_mean"].values
        d[x] = values[0] if values.size > 0 else None
    t = sorted(d.keys(), key=lambda x: d[x] if d[x] is not None else -1, reverse=True)
    grouped = grouped.sort_index(level=0, key=lambda x: [t.index(i) for i in x])
    grouped = grouped.reset_index()

    fig = px.line(
        grouped,
        x="concerned_year",
        y="all_mean",
        color="name",
        title=title,
        color_discrete_sequence=px.colors.qualitative.Light24_r,
        hover_data={"name": True, "concerned_year": True, "all_mean": True},
    )
    fig.update_traces(line=dict(width=3.5))  # Set line width to 3.5
    fig.update_layout(
        xaxis_title="År",
        yaxis_title="Score",
        autosize=False,
        width=fig_size[0],
        height=fig_size[1],
        plot_bgcolor="white",
        legend_title_text="Departementene, med synkende rekkefølge for året 2023",
    )
    fig.update_yaxes(showticklabels=False, title_text="")

    if print_output:
        # print how many elements are in each group
        g = df[df["name"].isin(names)].groupby(["name", "concerned_year"])
        g = g["top_k_mean"].count()
        old = pd.get_option("display.max_rows")
        pd.set_option("display.max_rows", None)
        print(g)
        pd.set_option("display.max_rows", old)

    return fig
