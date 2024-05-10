from .concerned_year import populate_concerned_year_ as populate_concerned_year
from .departments import (
    get_all_organisations,
    get_department,
    get_departments,
    get_directorates,
)
from .deviation_scaled_with_length import populate_deviation_scaled_with_length
from .get_documents_by import get_documents_by_name
from .lengths import (
    populate_length_of_split_paragraphs,
    populate_lengths,
    remove_empty_documents,
)
from .mean_sims import (
    DEFAULT_K,
    get_document_with_nth_highest_mean,
    populate_all_mean,
    populate_pure_mean_sims,
    populate_top_k_mean,
    populate_weighted_mean_sims,
)
from .sims import (
    adjust_sims_with_neighbors,
    populate_scaled_sims,
    populate_sims,
    return_paragraphs_with_highest_score,
)


def populate_everything(df, inn_text_embedding, k=DEFAULT_K):
    populate_lengths(df)
    remove_empty_documents(df)
    populate_concerned_year(df)
    populate_sims(df, inn_text_embedding)
    # adjust_sims_with_neighbors(df)
    populate_scaled_sims(df)
    populate_length_of_split_paragraphs(df)
    populate_deviation_scaled_with_length(df)
    populate_all_mean(df, k)
    return df
