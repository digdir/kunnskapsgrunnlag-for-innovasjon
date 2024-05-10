import textwrap

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter


def _set_target_score_and_get_plot_data_and_num_paras(
    df: pd.DataFrame, target_score: float
):
    max_unique_years = min(len(df["concerned_year"].unique()), 6)
    plot_data = []
    while (
        len(set(d["year"] for d in plot_data)) != max_unique_years
    ) and target_score > 0:
        num_paras_each_year_year = {}
        num_paras_each_year_grant = {}
        plot_data = []
        target_score -= 0.1
        for _, row in df.iterrows():
            year = row["concerned_year"]
            typerep = row["type"]
            for score in row["deviation_scaled_with_length"]:
                if typerep == "Årsrapport":
                    num_paras_each_year_year[year] = (
                        num_paras_each_year_year.get(year, 0) + 1
                    )
                elif typerep == "Tildelingsbrev":
                    num_paras_each_year_grant[year] = (
                        num_paras_each_year_grant.get(year, 0) + 1
                    )
                if score > target_score:
                    # add the year, score and type to the plot_data
                    plot_data.append({"year": year, "score": score, "type": typerep})
    plot_df = pd.DataFrame(plot_data)
    plot_df["year"] = plot_df["year"].astype(int)
    plot_df["score"] = plot_df["score"].astype(float)
    return target_score, plot_df, num_paras_each_year_year, num_paras_each_year_grant


def _get_yearrep_and_grants(
    plot_df: pd.DataFrame,
    num_paras_each_year_year: dict,
    num_paras_each_year_grant: dict,
):
    yearreps = plot_df[plot_df["type"] == "Årsrapport"]
    yearreps = yearreps.groupby(["year"]).size().reset_index(name="counts")
    yearreps["counts"] = yearreps["counts"] / yearreps["year"].apply(
        lambda x: num_paras_each_year_year[x]
    )

    grants = plot_df[plot_df["type"] == "Tildelingsbrev"]
    grants = grants.groupby(["year"]).size().reset_index(name="counts")
    grants["counts"] = grants["counts"] / grants["year"].apply(
        lambda x: num_paras_each_year_grant[x]
    )
    return yearreps, grants


def get_plotting_df(
    yearreps: pd.DataFrame,
    grants: pd.DataFrame,
    num_paras_each_year_year,
    num_paras_each_year_grant,
):
    fraction_unfiltered = pd.DataFrame(
        {
            "Tildelingsbrev, andel over terskel": grants.set_index("year")["counts"],
            "Årsrapporter, andel over terskel": yearreps.set_index("year")["counts"],
        }
    )
    fraction = fraction_unfiltered.dropna(axis=1, thresh=2)
    fraction = fraction.sort_index()

    num_paras_each_year_unfiltered = pd.DataFrame(
        {
            "Tildelingsbrev, antall paragrafer": pd.Series(num_paras_each_year_grant),
            "Årsrapporter, antall paragrafer": pd.Series(num_paras_each_year_year),
        }
    )
    num_paras_each_year = num_paras_each_year_unfiltered.dropna(axis=1, thresh=2)
    num_paras_each_year = num_paras_each_year.sort_index()

    return (
        fraction_unfiltered,
        fraction,
        num_paras_each_year_unfiltered,
        num_paras_each_year,
    )


def _plot(
    fraction_unfiltered,
    fraction,
    num_paras_each_year_unfiltered,
    num_paras_each_year,
    org,
    question,
    target_score,
):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax2 = ax.twinx()  # create a second y-axis
    if not fraction.empty:
        fraction.plot(kind="line", ax=ax)

    # Identify and plot points that only have one entry
    for column in fraction_unfiltered.columns:
        single_entry = fraction_unfiltered[column].dropna()
        if len(single_entry) == 1:
            ax.plot(
                single_entry.index, single_entry.values, "o", label=column, color="red"
            )

    # Plot the number of paragraphs on the secondary y-axis
    if not num_paras_each_year.empty:
        num_paras_each_year.plot(kind="line", ax=ax2, linestyle="--")

    # Identify and plot points that only have one entry
    for column in num_paras_each_year_unfiltered.columns:
        single_entry = num_paras_each_year_unfiltered[column].dropna()
        if len(single_entry) == 1:
            ax2.plot(
                single_entry.index,
                single_entry.values,
                "o",
                label=column,
                color="green",
            )

    ax2.set_ylabel("Antall paragrafer")  # set the label for the secondary y-axis

    # Get the legend handles and labels from both axes
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    # Combine the handles and labels
    handles = handles1 + handles2
    labels = labels1 + labels2

    # Create a dictionary with labels as keys and handles as values
    legend_dict = dict(zip(labels, handles))

    # Order the labels and handles
    ordered_labels = []
    # if it is in fraction or fraction unfiltered and has one entry
    if "Tildelingsbrev, andel over terskel" in fraction.columns or (
        "Tildelingsbrev, andel over terskel" in fraction_unfiltered.columns
        and len(fraction_unfiltered["Tildelingsbrev, andel over terskel"].dropna()) == 1
    ):
        ordered_labels.extend(
            [
                "Tildelingsbrev, andel over terskel",
                "Tildelingsbrev, antall paragrafer",
            ]
        )
    if "Årsrapporter, andel over terskel" in fraction.columns or (
        "Årsrapporter, andel over terskel" in fraction_unfiltered.columns
        and len(fraction_unfiltered["Årsrapporter, andel over terskel"].dropna()) == 1
    ):
        ordered_labels.extend(
            [
                "Årsrapporter, andel over terskel",
                "Årsrapporter, antall paragrafer",
            ]
        )
    ordered_handles = [legend_dict[label] for label in ordered_labels]

    # Create a single legend with the ordered handles and labels
    ax.legend(ordered_handles, ordered_labels, title="Dokumenttype")

    # dont create legend for ax2
    if ax2.get_legend() is not None:
        ax2.get_legend().remove()

    ax2.set_xticks([2018, 2019, 2020, 2021, 2022, 2023])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlim(2017.7, 2023.3)

    if org is not None:
        ax.set_title(f"{org}", fontsize=10)
    fig.suptitle("\n".join(textwrap.wrap(question, 100)))
    ax.set_xlabel("År")
    ax.set_ylabel(f"Andel tekstsnutter med en score over {target_score:.1f}")
    plt.close()
    return fig


def plot_frequency(
    df: pd.DataFrame, question: str, org=None, target_score: float = 0.9
) -> Figure:
    if org is not None:
        df = df[df["name"] == org]

    df = df[~df["concerned_year"].isin([2024, 2017])]
    target_score, plot_df, num_paras_each_year_year, num_paras_each_year_grant = (
        _set_target_score_and_get_plot_data_and_num_paras(df, target_score)
    )

    yearreps, grants = _get_yearrep_and_grants(
        plot_df, num_paras_each_year_year, num_paras_each_year_grant
    )

    # plot the dataframe with bar charts for each year
    # the y axis counts the occurences

    # Create a new DataFrame with yearreps and grants
    (
        fraction_unfiltered,
        fraction,
        num_paras_each_year_unfiltered,
        num_paras_each_year,
    ) = get_plotting_df(
        yearreps, grants, num_paras_each_year_year, num_paras_each_year_grant
    )

    return _plot(
        fraction_unfiltered,
        fraction,
        num_paras_each_year_unfiltered,
        num_paras_each_year,
        org,
        question,
        target_score,
    )
