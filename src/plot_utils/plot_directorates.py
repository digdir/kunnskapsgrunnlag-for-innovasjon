import textwrap

import pandas as pd
import plotly.express as px

from src.utils import get_department


def plot_directorates_yearly_score(df, directorate, num_bins=999):
    directory_data = df[(df["name"] == directorate) & (df.type == "Årsrapport")]

    plot_df = pd.DataFrame(columns=["year", "value", "text"])

    for year, data, text in zip(
        directory_data["concerned_year"],
        directory_data["deviation_scaled_with_length"],
        directory_data["split_paragraphs"],
    ):
        for j in range(len(data)):
            plot_df.loc[len(plot_df)] = [year, data[j], text[j][1]]

    plot_df["value_bin"] = pd.cut(plot_df["value"], bins=num_bins, labels=False)
    plot_df = plot_df.groupby(["year", "value_bin"]).first().reset_index()

    plot_df["text"] = plot_df["text"].apply(
        lambda x: "<br>".join(textwrap.wrap(x, width=70))
    )

    plot_df["text"] = plot_df["text"].fillna("No data available")

    fig = px.scatter(
        plot_df,
        x="year",
        y="value",
        hover_data="text",
        title=get_department(df, directorate),
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.0,
        y=1.05,
        text=directorate,
        showarrow=False,
        font=dict(size=12, color="black"),
    )

    fig.update_layout(xaxis_title="År", yaxis_title="Score", autosize=True, height=700)
    # Show the plot
    fig.update_traces(
        hovertemplate='<b>Score: %{y}<br>"%{customdata[0]}"</b><extra></extra>'
    )
    return fig
