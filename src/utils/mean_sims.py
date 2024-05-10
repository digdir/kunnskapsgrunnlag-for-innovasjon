import numpy as np
import pandas as pd

from .deviation_scaled_with_length import populate_deviation_scaled_with_length

DEFAULT_K = 100


def populate_weighted_mean_sims(df: pd.DataFrame) -> None:
    def weighted_mean(row):
        if len(row["sims_scaled"]) == 0:
            return 0
        return np.average(
            row["sims_scaled"],
            weights=row["length_of_split_paragraphs"],
        )

    df["weighted_mean_sims"] = df.apply(weighted_mean, axis=1)


def populate_pure_mean_sims(df: pd.DataFrame) -> None:
    df["pure_mean_sims"] = df["sims_scaled"].apply(
        lambda x: np.mean(x) if len(x) > 0 else 0
    )


# def _get_top_k(df: pd.DataFrame, k: int) -> list[str]:
#     return df["deviation_scaled_with_length"].apply(
#         lambda x: np.partition(x, -k)[-k:] if len(x) > k else x if len(x) > 0 else []
#     )


# def populate_top_k_mean(df: pd.DataFrame, k: int = DEFAULT_K) -> list[str]:
#     df["top_k_mean"] = _get_top_k(df, k).apply(
#         lambda x: np.mean(x) if len(x) > 0 else 0
#     )


def _get_top_k(df: pd.DataFrame, k: int) -> list[str]:
    combined_list = np.concatenate(df["deviation_scaled_with_length"].values)
    top_k = (
        np.partition(combined_list, -k)[-k:]
        if len(combined_list) > k
        else combined_list
        if len(combined_list) > 0
        else []
    )
    return top_k


def populate_top_k_mean(
    df: pd.DataFrame,
    k: int = DEFAULT_K,
    groupby: list[str] = ["name", "concerned_year"],
) -> list[str]:
    grouped = df.groupby(groupby)
    top_k_mean = grouped.apply(
        lambda x: np.mean(_get_top_k(x, k)) if len(_get_top_k(x, k)) > 0 else 0
    ).reset_index()
    df["top_k_mean"] = df.merge(top_k_mean, on=groupby, how="left")[0]


def get_document_with_nth_highest_mean(
    df: pd.DataFrame, n: int, weighted: bool = True
) -> pd.Series:
    max_idx = df["weighted_mean_sims" if weighted else "pure_mean_sims"].values
    second_highest = np.partition(max_idx, -n)[-n]
    idx = np.where(max_idx == second_highest)[0][0]
    return df.loc[idx]


def populate_all_mean(
    df: pd.DataFrame,
    k: int = DEFAULT_K,
    groupby: list[str] | str = ["name", "concerned_year"],
) -> None:
    populate_weighted_mean_sims(df)
    weight_mean = df.groupby(groupby)["weighted_mean_sims"].transform("mean")
    populate_pure_mean_sims(df)
    mean = df.groupby(groupby)["pure_mean_sims"].transform("mean")
    populate_top_k_mean(df, k, groupby)
    top_k = df.groupby(groupby)["top_k_mean"].transform("mean")
    populate_deviation_scaled_with_length(df)
    weight_dev = df["deviation_scaled_with_length"].apply(
        lambda x: np.mean(x) if len(x) > 0 else 0
    )
    df["all_mean"] = (top_k + mean + weight_mean + weight_dev) / 4
