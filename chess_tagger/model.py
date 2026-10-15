"""Inference for the fine-tuned chess tagger (v3: mentions, move attributes, claims).

The model is the LFM2.5 encoder plus ONE linear layer whose outputs are split
into four heads, each with its own softmax:

    mention  BIO over MOVE, PIECE, SQUARE, FILE, RANK, DIAGONAL   (13 labels)
    line     NEW | NEXT | ALT                  (read on the tokens of a MOVE span)
    stance   CHOSEN | REJECTED | CONSIDERED    (read on the tokens of a MOVE span)
    frame    NOW | AFTER                       (read on the tokens of a PIECE span or a claim)
    claim    BIO over CHECK, MATE, WINS, ATTACKS, FLIGHT          (11 labels)

scripts/train.py trains it; keep the decoding rules here and in web/tagger.js in sync.

    tagger = ChessTagger("chess-move-tagger")
    tagger.tag("The queen slides to h8 delivering mate, as the king on h7 cannot go to g8.")
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from .markup import CLAIM_TYPES, FRAMES, LINES, MENTION_TYPES, STANCES, MarkupError, render

TRIM = " \n\t.,;:!?*`()\"'"  # never at the edge of a span (but + # = - are kept)


def bio(types):
    return ["O"] + [f"{prefix}-{kind}" for kind in types for prefix in "BI"]


HEADS = {"mention": bio(MENTION_TYPES), "line": LINES, "stance": STANCES, "frame": FRAMES,
         "claim": bio(CLAIM_TYPES)}


def head_slices(heads: dict = HEADS) -> dict:
    out, offset = {}, 0
    for name, labels in heads.items():
        out[name] = slice(offset, offset + len(labels))
        offset += len(labels)
    return out


NUM_OUTPUTS = sum(len(v) for v in HEADS.values())


class ChessMoveTagger(nn.Module):
    """Pretrained LFM2.5 encoder body + one linear layer holding all four heads."""

    def __init__(self, encoder: nn.Module, num_labels: int = NUM_OUTPUTS, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.classifier(self.dropout(hidden))


def bio_spans(text: str, offsets: list, labels: list[str]) -> list:
    """One BIO label per token -> [[start, end, type, first_token, last_token], ...]."""
    spans, inside = [], False
    for t, ((start, end), label) in enumerate(zip(offsets, labels)):
        while start < end and text[start].isspace():
            start += 1
        if start == end or label == "O":
            inside = False
            continue
        tag, kind = label.split("-", 1)
        mid_word = spans and spans[-1][1] == start and start > 0 and text[start - 1].isalnum() and text[start].isalnum()
        if mid_word or (inside and tag == "I" and kind == spans[-1][2]):
            spans[-1][1], spans[-1][4] = end, t  # a word is never split between two spans: the first token decides
        else:
            spans.append([start, end, kind, t, t])
        inside = True
    cleaned = []
    for start, end, kind, t0, t1 in spans:
        while start < end and text[start] in TRIM:
            start += 1
        while end > start and text[end - 1] in TRIM:
            end -= 1
        if start < end:
            cleaned.append([start, end, kind, t0, t1])
    return cleaned


def decode(text: str, offsets: list, logits: torch.Tensor) -> dict:
    """Token logits [tokens, NUM_OUTPUTS] ->
    {"mentions": [[s, e, type, line, stance, frame]], "claims": [[s, e, type, frame]]} (None where n/a)."""
    sl = head_slices()

    def attribute(head, labels, t0, t1):  # average the head's log-probabilities over the span's tokens
        return labels[int(logits[t0:t1 + 1, sl[head]].log_softmax(-1).mean(0).argmax())]

    mention_ids = logits[:, sl["mention"]].argmax(-1).tolist()
    claim_ids = logits[:, sl["claim"]].argmax(-1).tolist()
    mentions = []
    for start, end, kind, t0, t1 in bio_spans(text, offsets, [HEADS["mention"][i] for i in mention_ids]):
        line = stance = frame = None
        if kind == "MOVE":
            line, stance = attribute("line", LINES, t0, t1), attribute("stance", STANCES, t0, t1)
        elif kind == "PIECE":
            frame = attribute("frame", FRAMES, t0, t1)
        mentions.append([start, end, kind, line, stance, frame])
    first_move = next((m for m in mentions if m[2] == "MOVE"), None)
    if first_move is not None:
        first_move[3] = "NEW"  # nothing to continue yet
    claims = [[s, e, k, attribute("frame", FRAMES, t0, t1)]
              for s, e, k, t0, t1 in bio_spans(text, offsets, [HEADS["claim"][i] for i in claim_ids])]
    return {"mentions": mentions, "claims": claims}


class ChessTagger:
    def __init__(self, model_dir: str | Path, device: str | None = None):
        model_dir = Path(model_dir)
        config = json.loads((model_dir / "tagger_config.json").read_text())
        if config.get("heads") != HEADS:
            raise ValueError("this model was trained with other labels; use the matching version of chess_tagger")
        self.max_length = config["max_length"]
        self.base_model = config["base_model"]
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else
                                              "mps" if torch.backends.mps.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.tokenizer.padding_side = "right"
        self.model = self._load(model_dir / "model.safetensors").to(self.device).eval()

    def _load(self, weights: Path) -> ChessMoveTagger:
        # Build the architecture from the base model's config only (no pretrained
        # weights download), then copy our weights in one tensor at a time.
        body_class = get_class_from_dynamic_module(
            "modeling_lfm2_bidirectional.Lfm2BidirectionalModel", self.base_model)
        body_config = AutoConfig.from_pretrained(self.base_model, trust_remote_code=True)
        model = ChessMoveTagger(body_class(body_config))
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
    def predict(self, texts: list[str], batch_size: int = 16) -> list[dict]:
        results = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            enc = self.tokenizer(chunk, return_offsets_mapping=True, padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
            logits = self.model(enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)).float().cpu()
            for j, (text, offsets) in enumerate(zip(chunk, enc["offset_mapping"].tolist())):
                n = int(enc["attention_mask"][j].sum())
                results.append(decode(text, offsets[:n], logits[j, :n]))
        return results

    def tag(self, text: str) -> dict:
        """Mentions and claims, plus the views the demo app uses: move series and board highlights."""
        n_tokens = len(self.tokenizer(text)["input_ids"])
        predicted = self.predict([text])[0]
        return {**group(text, predicted), "truncated": n_tokens > self.max_length}


def group(text: str, predicted: dict) -> dict:
    mentions, series, highlights = [], [], []
    for start, end, kind, line, stance, frame in predicted["mentions"]:
        item = {"start": start, "end": end, "type": kind, "text": text[start:end]}
        if kind == "MOVE":
            # A series is what the app replays: NEW starts one; NEXT extends it; ALT replaces its last move.
            if line == "NEW" or not series:
                series.append([])
            elif line == "ALT" and series[-1]:
                series.append(series[-1][:-1])
            item.update(line=line, stance=stance, series=len(series) - 1, index=len(series[-1]))
            series[-1].append(item)
        else:
            if kind == "PIECE":
                item["frame"] = frame
            highlights.append(item)
        mentions.append(item)
    claims = [{"start": s, "end": e, "type": k, "frame": f, "text": text[s:e]} for s, e, k, f in predicted["claims"]]
    try:
        marked = render(text, predicted["mentions"], predicted["claims"])
    except MarkupError:  # a claim overlapping a mention (the heads are independent)
        marked = None
    return {"mentions": mentions, "claims": claims, "series": series, "highlights": highlights, "marked": marked}
