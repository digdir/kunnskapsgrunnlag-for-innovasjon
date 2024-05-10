from typing import Generator

from .chunker import (
    parse_and_split_paragraphs_on_max_length_on_sentence,
)
from .model import EmbeddingModel
from .tokenizer import EmbeddingTokenizer


def embed_plaintext_for_document(
    plaintext: str, model: EmbeddingModel, tokenizer: EmbeddingTokenizer
) -> Generator[tuple, None, None]:
    paragraphs = [
        x
        for x in parse_and_split_paragraphs_on_max_length_on_sentence(
            plaintext, tokenizer
        )
    ]
    embeddings = []
    for i in range(0, len(paragraphs), 1):
        embeddings.append(model(tokenizer(paragraphs[i : i + 1])))
    return paragraphs, embeddings
