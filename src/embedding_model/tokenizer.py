from transformers import AutoTokenizer, BatchEncoding

AVAILABLE_MODELS = ["intfloat/multilingual-e5-small"]


class EmbeddingTokenizer:
    def __init__(self, pretrained_path: str = "intfloat/multilingual-e5-small") -> None:
        self.tokenizer = None
        if pretrained_path not in AVAILABLE_MODELS:
            raise ValueError(f"Model {pretrained_path} not available")
        self.pretrained_path = pretrained_path

    def load(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_path)

    def len(
        self,
        input_text: str,
        *,
        prefix: str = "query: ",
    ) -> int:
        assert self.tokenizer is not None, "Tokenizer not loaded"

        v = self.tokenizer(f"{prefix}{input_text}", return_length=True).get(
            "length", []
        )
        if len(v) < 0:
            raise ValueError("No length from tokenizer")
        return int(v[0])

    def __call__(
        self,
        input_texts: list[str],
        *,
        prefix: str = "query: ",
    ) -> BatchEncoding:
        assert self.tokenizer is not None, "Tokenizer not loaded"
        assert isinstance(input_texts, list), "Input must be a list of strings"
        return self.tokenizer(
            [f"{prefix}{t}" for t in input_texts],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
