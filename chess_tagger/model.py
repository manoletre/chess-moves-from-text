"""Inference for the fine-tuned chess move & board-location tagger.

The model architecture and the decoding rules mirror
notebooks/train_chess_move_tagger.ipynb; keep them in sync.

    tagger = ChessTagger("chess-move-tagger")
    tagger.tag("The game continued Nb8 Nf3 c6, hitting d5.")
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

SPAN_TYPES = ["START", "CONT", "SQUARE", "FILE", "RANK", "DIAGONAL"]
MOVE_TYPES = {"START", "CONT"}
TRIM = " \n\t.,;:!?*`()\"'"  # never part of a move or location (but + # = - are)


class ChessMoveTagger(nn.Module):
    """Pretrained LFM2.5 encoder body + a linear token-classification head."""

    def __init__(self, encoder: nn.Module, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.classifier(self.dropout(hidden))


def decode(text: str, offsets: list, label_ids: list, id2label: dict) -> list:
    """One predicted label per token -> [[start, end, type], ...] character spans."""
    spans, inside_span = [], False
    for (start, end), label_id in zip(offsets, label_ids):
        while start < end and text[start].isspace():
            start += 1
        label = id2label[label_id]
        if start == end or label == "O":
            inside_span = False
            continue
        tag, kind = label.split("-")
        if tag == "B" or not inside_span:
            spans.append([start, end, kind])
        else:
            spans[-1][1] = end
        inside_span = True
    cleaned = []
    for start, end, kind in spans:
        while start < end and text[start] in TRIM:
            start += 1
        while end > start and text[end - 1] in TRIM:
            end -= 1
        if start < end:
            cleaned.append([start, end, kind])
    return cleaned


class ChessTagger:
    def __init__(self, model_dir: str | Path, device: str | None = None):
        model_dir = Path(model_dir)
        config = json.loads((model_dir / "tagger_config.json").read_text())
        self.labels = config["labels"]
        self.id2label = dict(enumerate(self.labels))
        self.max_length = config["max_length"]
        self.base_model = config["base_model"]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.tokenizer.padding_side = "right"
        self.model = self._load(model_dir / "model.safetensors").to(self.device).eval()

    def _load(self, weights: Path) -> ChessMoveTagger:
        # Build the architecture from the base model's config only (no pretrained
        # weights download), then copy our weights in one tensor at a time. Peak
        # memory stays ~1x the model size instead of 2x.
        body_class = get_class_from_dynamic_module(
            "modeling_lfm2_bidirectional.Lfm2BidirectionalModel", self.base_model)
        body_config = AutoConfig.from_pretrained(self.base_model, trust_remote_code=True)
        model = ChessMoveTagger(body_class(body_config), num_labels=len(self.labels))
        params = model.state_dict()
        with safe_open(str(weights), framework="pt") as f:
            keys = set(f.keys())
            missing, unexpected = set(params) - keys, keys - set(params)
            if missing or unexpected:
                raise ValueError(f"weights don't match the model: missing={sorted(missing)[:5]} "
                                 f"unexpected={sorted(unexpected)[:5]}")
            with torch.no_grad():
                for key in keys:
                    params[key].copy_(f.get_tensor(key))
        return model

    @torch.no_grad()
    def predict_spans(self, texts: list[str], batch_size: int = 16) -> list[list]:
        results = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = self.tokenizer(chunk, return_offsets_mapping=True, padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
            logits = self.model(enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device))
            predictions = logits.argmax(-1).tolist()
            for text, offsets, label_ids in zip(chunk, enc["offset_mapping"].tolist(), predictions):
                results.append(decode(text, offsets, label_ids, self.id2label))
        return results

    def tag(self, text: str) -> dict:
        """Spans plus the two views chess-tutor needs: move series and board highlights."""
        n_tokens = len(self.tokenizer(text)["input_ids"])
        spans, series, highlights = [], [], []
        for start, end, kind in self.predict_spans([text])[0]:
            item = {"start": start, "end": end, "type": kind, "text": text[start:end]}
            if kind in MOVE_TYPES:
                if kind == "START" or not series:
                    series.append([])
                item["series"], item["index"] = len(series) - 1, len(series[-1])
                series[-1].append(item)
            else:
                highlights.append(item)
            spans.append(item)
        return {"spans": spans, "series": series, "highlights": highlights,
                "truncated": n_tokens > self.max_length}
