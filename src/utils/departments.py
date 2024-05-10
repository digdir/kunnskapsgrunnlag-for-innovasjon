from typing import Dict, List, Union

import pandas as pd


def get_departments(df: pd.DataFrame) -> list[str]:
    series = df[df.level == 0]["name"].value_counts()
    labels_to_drop = [
        "STORTINGET",
        "SAMEDIGGI / SAMETINGET",
        "STATSMINISTERENS KONTOR",
        "DIGITALISERINGS- OG FORVALTNINGSDEPARTEMENTET (DFD)",
    ]
    labels_to_drop = [label for label in labels_to_drop if label in series.index]
    return list(series.drop(labels_to_drop).to_dict())


def get_department(df: pd.DataFrame, directorate: str) -> str:
    """
    Returns the department corresponding to the given directorate.
    """

    # Find the row in the dataframe where the name is the directorate
    directorate_row = df[df.name == directorate]

    # If the directorate is not found, return an empty string
    if directorate_row.empty:
        return ""

    # Get the parent_id of the directorate
    parent_id = directorate_row["parent_id"].values[0]

    # Find the department that has this id
    department_row = df[df["id.4"] == parent_id]

    # If the department is not found, return an empty string
    if department_row.empty:
        return ""

    # Return the name of the department
    return department_row["name"].values[0]


def get_directorates(
    df: pd.DataFrame, departments: Union[str, List[str]] = []
) -> Union[List[str], Dict[str, List[str]]]:
    """
    Returns the directorates corresponding to the given departments.
    - If no departments are given, all departments are considered.
    - If only one department is given, the function returns a list of directorates.
    - If multiple departments are given, the function returns a dictionary where the keys
        are the departments and the values are the corresponding directorates.
    """

    if isinstance(departments, str):
        departments = [departments]

    if len(departments) == 0:
        departments = get_departments(df)

    directorates = {}
    for department in departments:
        value = df[df.name == department]["id.4"].values[0]
        directorates[department] = list(df[df.parent_id == value].name.unique())

    if len(directorates) == 1:
        return list(directorates.values())[0]

    return directorates


def get_all_organisations(df: pd.DataFrame) -> list[str]:
    return list(df["name"].unique())
