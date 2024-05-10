import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, BatchEncoding

AVAILABLE_MODELS = ["intfloat/multilingual-e5-small"]


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class EmbeddingModel:
    def __init__(self, pretrained_path: str = "intfloat/multilingual-e5-small") -> None:
        self.model = None
        if pretrained_path not in AVAILABLE_MODELS:
            raise ValueError(f"Model {pretrained_path} not available")
        self.pretrained_path = pretrained_path

    def load(self):
        if self.model is None:
            self.model = AutoModel.from_pretrained(self.pretrained_path)

    def __call__(self, batch_dict: BatchEncoding) -> list[list[float]]:
        assert self.model is not None, "Model is not loaded"

        if self.pretrained_path == "intfloat/multilingual-e5-small":
            outputs = self.model(**batch_dict)
            batch_embeds = average_pool(
                outputs.last_hidden_state,
                batch_dict.get("attention_mask", None),  # type: ignore
            )
            batch_embeds = F.normalize(batch_embeds, p=2, dim=1)
            return batch_embeds.tolist()
        raise NotImplementedError(f"Model {self.pretrained_path} not callable")
