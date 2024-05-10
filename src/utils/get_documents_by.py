import pandas as pd


def get_documents_by_name(df, name):
    """
    Gitt en virksomhet, henter både de årsrapportene de selv har laget, og tildelingsbrevene de har mottatt.
    """

    data = df[(df["name"] == name) & (df.type == "Årsrapport")]
    # print(data["actor_id"])
    if "recipient_id" in df.columns:
        filtered_data = df[df["recipient_id"] == data["actor_id"].iloc[0]]
        return pd.concat([data, filtered_data])

    print("Warning: recipient_id not in dataframe, only fetching 'Årsrapport'")
    return data
