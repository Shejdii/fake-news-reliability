import torch

from src.confidence.distilbert_confidence import predict_with_confidence_distilbert


class DummyTokenizer:
    def __call__(self, text, truncation=True, max_length=256, return_tensors="pt"):
        return {
            "input_ids": torch.tensor([[101, 2023, 2003, 102]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        }


class DummyOutput:
    def __init__(self):
        self.logits = torch.tensor([[0.1, 0.2, 2.5]])


class DummyDistilBertModel:
    def eval(self):
        return self

    def __call__(self, **kwargs):
        return DummyOutput()


def test_predict_with_confidence_distilbert():
    model = DummyDistilBertModel()
    tokenizer = DummyTokenizer()

    result = predict_with_confidence_distilbert(
        model=model,
        tokenizer=tokenizer,
        text="Example claim.",
    )

    assert result["predicted_label"] == "TRUE"
    assert result["confidence_band"] in {"fairly_sure", "very_sure"}
    assert result["confidence_score"] > 0.0
