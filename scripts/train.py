"""Train the v3 tagger (one encoder, one linear layer split into five heads).

    python scripts/train.py                      # data/{train,val}.jsonl -> chess-move-tagger/
    python scripts/train.py --epochs 1 --limit 32  # smoke test

Runs on CUDA, Apple MPS or CPU. See notebooks/train_chess_move_tagger.ipynb for a
step-by-step explanation of token classification; this script is the same recipe
with several heads, each with its own cross-entropy:

    mention, claim   BIO labels on every token
    line, stance     only on the tokens of MOVE spans
    frame            only on the tokens of PIECE spans and claims

Texts from the old commentary data have no stance/frame/claim labels; those heads
get -100 (ignored) on them instead of being taught "nothing here".
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_model
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer, get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from chess_tagger.markup import FRAMES, LINES, STANCES  # noqa: E402
from chess_tagger.metrics import headline, score, summary  # noqa: E402
from chess_tagger.model import BASE_REVISION, HEADS, ChessMoveTagger, decode, head_slices  # noqa: E402

MODEL_ID = "LiquidAI/LFM2.5-Encoder-230M"
MODEL_REVISION = BASE_REVISION  # pinned Hub commit: weights, tokenizer and remote code
MAX_LENGTH = 512
IGNORE = -100


def token_labels(text: str, offsets: list, example: dict) -> dict:
    full = example.get("labelled") == "full"
    labels = {name: [] for name in HEADS}
    mention_index = {lab: i for i, lab in enumerate(HEADS["mention"])}
    claim_index = {lab: i for i, lab in enumerate(HEADS["claim"])}
    for start, end in offsets:
        if start == end:
            for name in HEADS:
                labels[name].append(IGNORE)
            continue
        while start < end and text[start].isspace():
            start += 1
        mention, line, stance, frame = "O", IGNORE, IGNORE, IGNORE
        for s, e, kind, m_line, m_stance, m_frame in example["mentions"]:
            if start < e and end > s:
                mention = ("B-" if start <= s else "I-") + kind
                if kind == "MOVE":
                    line = LINES.index(m_line)
                    if full:
                        stance = STANCES.index(m_stance)
                elif kind == "PIECE" and full and m_frame:
                    frame = FRAMES.index(m_frame)
                break
        claim = "O"
        for s, e, kind, c_frame in example.get("claims") or []:
            if start < e and end > s:
                claim = ("B-" if start <= s else "I-") + kind
                if c_frame:
                    frame = FRAMES.index(c_frame)
                break
        labels["mention"].append(mention_index[mention])
        labels["line"].append(line)
        labels["stance"].append(stance)
        labels["frame"].append(frame)
        labels["claim"].append(claim_index[claim] if full else IGNORE)
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=ROOT / "data", type=Path)
    parser.add_argument("--out", default=ROOT / "chess-move-tagger", type=Path)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--oversample", type=int, default=2, help="repeat fully labelled texts this many times")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    load = lambda name: [json.loads(line) for line in (args.data / f"{name}.jsonl").read_text().splitlines()]
    train, val = load("train"), load("val")
    if args.limit:
        train, val = train[:args.limit], val[:args.limit // 2]
    val_refl = [e for e in val if e["group"] == "reflection"] or val
    print(f"device {device} | train {len(train)} | val {len(val)} ({len(val_refl)} reflections)")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True)
    tokenizer.padding_side = "right"

    def encode(example):
        enc = tokenizer(example["text"], return_offsets_mapping=True, truncation=True, max_length=MAX_LENGTH)
        return {"input_ids": enc["input_ids"], **token_labels(example["text"], enc["offset_mapping"], example)}

    features = []
    for example in train:
        f = encode(example)
        features += [f] * (args.oversample if example.get("labelled") == "full" else 1)

    def collate(batch):
        width = max(len(b["input_ids"]) for b in batch)
        pad = lambda v, x: v + [x] * (width - len(v))
        out = {"input_ids": torch.tensor([pad(b["input_ids"], tokenizer.pad_token_id) for b in batch]),
               "attention_mask": torch.tensor([pad([1] * len(b["input_ids"]), 0) for b in batch])}
        for name in HEADS:
            out[name] = torch.tensor([pad(b[name], IGNORE) for b in batch])
        return out

    loader = DataLoader(features, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    encoder = AutoModelForMaskedLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, code_revision=MODEL_REVISION, trust_remote_code=True).lfm2
    model = ChessMoveTagger(encoder).to(device)
    slices = head_slices()

    @torch.no_grad()
    def predict(examples, batch_size=32):
        model.eval()
        results = []
        for i in range(0, len(examples), batch_size):
            texts = [e["text"] for e in examples[i:i + batch_size]]
            enc = tokenizer(texts, return_offsets_mapping=True, padding=True, truncation=True,
                            max_length=MAX_LENGTH, return_tensors="pt")
            logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device)).float().cpu()
            for j, (text, offsets) in enumerate(zip(texts, enc["offset_mapping"].tolist())):
                n = int(enc["attention_mask"][j].sum())
                results.append(decode(text, offsets[:n], logits[j, :n]))
        model.train()
        return results

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    total = args.epochs * len(loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total), total)
    best, history = -1.0, []
    model.train()
    for epoch in range(1, args.epochs + 1):
        started, losses = time.time(), []
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = 0.0
            for name in HEADS:
                target = batch[name]
                if (target != IGNORE).any():
                    head = logits[..., slices[name]]
                    loss = loss + F.cross_entropy(head.reshape(-1, head.size(-1)), target.reshape(-1), ignore_index=IGNORE)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        metrics = score(val_refl, predict(val_refl))
        all_val = score(val, predict(val))
        h = headline(metrics)
        history.append({"epoch": epoch, "loss": sum(losses) / len(losses), "headline": h, "val_reflections": metrics})
        saved = h > best
        if saved:
            best = h
            args.out.mkdir(parents=True, exist_ok=True)
            save_model(model, str(args.out / "model.safetensors"))
            tokenizer.save_pretrained(args.out)
            (args.out / "tagger_config.json").write_text(json.dumps(
                {"base_model": MODEL_ID, "base_revision": MODEL_REVISION, "heads": HEADS, "max_length": MAX_LENGTH,
                 "version": "3.0.0"}, indent=2))
        print(f"epoch {epoch} ({time.time() - started:.0f}s) loss {history[-1]['loss']:.4f} | "
              f"val reflections: {summary(metrics)} | all val mention F1 {all_val['mention']['f1']:.3f}"
              + ("  ✓ saved" if saved else ""), flush=True)
    (args.out / "training_history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
