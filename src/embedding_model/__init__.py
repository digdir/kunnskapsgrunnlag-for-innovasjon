from .model import EmbeddingModel as EmbeddingModel
from .tokenizer import EmbeddingTokenizer as EmbeddingTokenizer


def embed(text: str):
    tokenizer = EmbeddingTokenizer()
    model = EmbeddingModel()
    tokenizer.load()
    model.load()
    return model(tokenizer([text]))[0]
