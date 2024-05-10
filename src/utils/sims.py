import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.preprocessing import MinMaxScaler


def populate_sims(df: pd.DataFrame, inn_text_embedding) -> None:
    df["sims"] = df["split_paragraphs"].apply(
        lambda x: [cosine(inn_text_embedding, y[2][0]) for y in x]
    )


def adjust_sims_with_neighbors(df: pd.DataFrame) -> None:
    def adjust_list(lst):
        new_lst = []
        if len(lst) == 1:
            return lst
        for i in range(len(lst)):
            if i == 0:
                new_lst.append(0.9 * lst[i] + 0.1 * lst[i + 1])
            elif i == len(lst) - 1:
                new_lst.append(0.9 * lst[i] + 0.1 * lst[i - 1])
            else:
                new_lst.append(0.9 * lst[i] + 0.05 * (lst[i - 1] + lst[i + 1]))
        return new_lst

    df["sims"] = df["sims"].apply(adjust_list)


def populate_scaled_sims(df: pd.DataFrame) -> None:
    scaler = MinMaxScaler()
    # combine all the arrays in df['sims'] into one array to fit the scaler
    sims = np.concatenate(df["sims"].values).reshape(-1, 1)
    scaler.fit(sims)
    df["sims_scaled"] = df["sims"].apply(
        lambda x: (
            1 - scaler.transform(np.array(x).reshape(-1, 1)).flatten()
            if len(x) > 0
            else []
        )
    )


def return_paragraphs_with_highest_score(df: pd.DataFrame, sim_column: str, k: int):
    # Create a list to store tuples of (paragraph, score)
    paragraphs_scores = []

    # Iterate over the DataFrame
    for _, row in df.iterrows():
        paras = row["split_paragraphs"]
        sims = row[sim_column]
        name = row["name"]
        title = row["title"]
        # Add all (paragraph, score) tuples to the list
        paragraphs_scores.extend(
            zip(paras, sims, [name] * len(paras), [title] * len(paras))
        )

    # Sort the list by score in descending order
    paragraphs_scores.sort(key=lambda x: x[1], reverse=True)

    # Print the top k paragraphs and their scores
    returned_texts = []
    returned_orgs = []
    return_str = ""
    i = 0
    while len(return_str.split("\n\n")) <= k and len(paragraphs_scores) > i:
        i += 1
        if (
            paragraphs_scores[i][0][1] in returned_texts
            or paragraphs_scores[i][2] in returned_orgs
        ):
            continue
        returned_orgs.append(paragraphs_scores[i][2])
        returned_texts.append(paragraphs_scores[i][0][1])
        return_str += f"Virksomhet: {paragraphs_scores[i][2]}\n"
        return_str += f"Dokument: {paragraphs_scores[i][3]}\n"
        return_str += f"Score: {paragraphs_scores[i][1]:.3f}\n"
        return_str += f"tekstsnutt: {paragraphs_scores[i][0][1]}\n\n"
    return return_str
