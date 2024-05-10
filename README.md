# DIGDIR-embedding-av-offentlige-dokumenter

This repository was used for the development of a solution that used word embeddings to gain insight from public documents. A lot of the code is written in a way to make it easily applicable to other projects with other question and other data. The embedding model multilingual-e5-small was used. The term for this method of embedding a text and comparing similarity in an embedding space is a [RAG](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/) method.

## Table of Contents

- [Table of Contents](#table-of-contents)
- [How to run](#how-to-run)
  - [If you don't want to use the database](#if-you-dont-want-to-use-the-database)
    - [Requirements](#requirements)
    - [Steps](#steps)
  - [If you want to use the database](#if-you-want-to-use-the-database)
    - [Requirements](#requirements-1)
    - [Steps](#steps-1)
- [What to change for other use cases](#what-to-change-for-other-use-cases)
- [Further development opportunities](#further-development-opportunities)

## How to run

### If you don't want to use the database

#### Requirements

- Python version 3.11 or higher. We recommend using [pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation) to manage your versions.
- [Poetry](https://python-poetry.org/docs/#installing-with-pipx)

#### Steps

1. Clone and `cd` into the repository
2. Run `poetry install` to install the dependencies. You're now ready to run code

*Note: Poetry installs a virtual environment with all of your packages. This means that in order to run the `python` command in the CLI, you need to first activate the venv by running `poetry shell`.*

### If you want to use the database

#### Requirements

To get started you will need the following software:

- `Docker`. For example [`Docker Desktop`](https://www.docker.com/products/docker-desktop/)
- [`VSCode`](https://code.visualstudio.com/) and the following extension-pack:
  - [`Remote Development`](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack). This is used to connect to the running `docker`-image.
- The [Kudos database .sql file](https://kudos.dfo.no/kudos-database-latest.sql) downloaded.
- **IF** developing on a `Windows`-installation:
  - [`WSL`](https://docs.microsoft.com/en-us/windows/wsl/install) to open the repo in and develop from. It also improves `docker` build performance.
  - The [`WSL`](https://code.visualstudio.com/docs/remote/wsl) vscode extension. Activate WSL in vscode before going forward to the next steps.

#### Steps

1. Clone and cd into the repository on your UNIX-based filesystem (`WSL`/`MacOS`/`Linux`).
2. Place your downloaded Kudos database file in the `/db` folder.
3. Open the `Command Palette` (shortcut: `ctrl + shift + p` / `cmd + shift + p`) inside `vscode`. Type and select `Dev Containers: Reopen in Container`. Your `dev container` should now start initializing. This will take a while, but you are ready to run code after this process is done. If you get an error with connecting to the database in the first 10 minutes after building, wait for a bit and try again, otherwise try entering the container database container to see if the `MariaDB` service is operational.

## What to change for other use cases

Anything in the `src/embedding_model` folder does not need to be changed in order to use this code for other use cases. It contains the code for the embedding model, and is not dependent on the data used.

For all of the other code, it depends on what data is used. If all of the same columns with the same data types are used, the code should be able to run without issue. The following is an explanation for all of the columns in the DataFrame for the documents that needs to be in the source DataFrame:

- `split_paragraphs`: a column where each element is list[tuple[int, str, np.array]]. The first index is the page number that the paragraph is from, the second index is the paragraph itself, and the third index is the embedding of the paragraph.
- `name`: the name of the organisation owning the document
- `title`: the title of the document
- `type`: the type of the document. For our use case this was either `"Årsrapport"` or `"Tildelingsbrev"`, but can easily be changed`
- `concerned_year`: which year is the document connected to. If this is not filled, `published_at` is needed
- `id.4` and `parent_id`: these are somewhat optional, and is used for connecting directorates with their parent departments

The columns that are populated by the code in the `utils` folder are:

- `length_of_split_paragraphs`: the character length of each paragraph in `split_paragraphs`. For each row, the length of the list corresponds to the length of the list in `split_paragraphs`
- `lengths`: the length of the list in `split_paragraphs`
- `sims`: the cosine similarity of the document to the query
- `sims_scaled`: `sims` scaled with the `MinMaxScaler` from `sklearn`
- `pure_mean_sims`: the unweightedmean of `sims_scaled`
- `weighted_mean_sims`: the mean of `sims_scaled` with  `length_of_split_paragraphs` as the weights
- `deviation_scaled_with_length`: A custom function to weight the deviation of the score of the paragraph with the length of the paragraph
- `top_k_mean`: the mean of the top k `deviation_scaled_with_length` values
- `all_mean`: the mean of the score of `top_k_mean`, `pure_mean_sims`, `weighted_mean_sims`, and `deviation_scaled_with_length`

## Further development opportunities

- Perhaps the easiest way to improve the performance would be to use a more powerful model. The model used in this project was `multilingual-e5-small`. A larger model, like `multilingual-e5-base` or `multilingual-e5-large` would likely give better results, but would also require more computational power.
- As stated previously, the premise of this project was to use word embeddings to gain insight from public documents. This could be expanded to include more documents, or to include documents from other sources.
- The code could be further optimized to run faster. This includes, but is not limited to, methods to reduce embedding time, reduce the time needed for finding the most similar documents, or by running the code on a more specialised machines.
