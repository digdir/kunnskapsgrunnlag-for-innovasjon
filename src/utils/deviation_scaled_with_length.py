import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

ALPHA = 5


def populate_deviation_scaled_with_length(df: pd.DataFrame):
    """
    This function populates the deviation_scaled_with_length column in the dataframe.
    `deviation_scaled_with_length` is calculated as follows:
    deviation_scaled_with_length = (similarity - mean_similarity) * (abs(length - 50))^(1/ALPHA)

    It is used to calculate the deviation of similarity scores from the mean similarity score in order to give more weight to the similarity scores of paragraphs with longer lengths.
    """
    mean_sims_scaled = np.mean(
        df["sims_scaled"].apply(lambda x: np.mean(x) if len(x) > 0 else 0)
    )

    df["deviation_scaled_with_length"] = [
        [(s - mean_sims_scaled) * (np.abs(l - 50)) ** (1 / ALPHA) for s, l in zip(x, y)]
        for x, y in zip(df["sims_scaled"], df["length_of_split_paragraphs"])
    ]
    scaler = MinMaxScaler()
    sims = np.concatenate(df["deviation_scaled_with_length"].values).reshape(-1, 1)
    scaler.fit(sims)
    df["deviation_scaled_with_length"] = df["deviation_scaled_with_length"].apply(
        lambda x: (
            scaler.transform(np.array(x).reshape(-1, 1)).flatten() if len(x) > 0 else []
        )
    )
