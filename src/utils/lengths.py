import pandas as pd


def populate_lengths(df: pd.DataFrame) -> None:
    df["lengths"] = df["split_paragraphs"].apply(lambda x: len(x))


def remove_empty_documents(df: pd.DataFrame) -> None:
    mask = df["lengths"] != 0
    df.drop(df[~mask].index, inplace=True)
    df.reset_index(drop=True, inplace=True)


def populate_length_of_split_paragraphs(df: pd.DataFrame) -> None:
    df["length_of_split_paragraphs"] = df["split_paragraphs"].apply(
        lambda l: [len(x[1]) for x in l]
    )
