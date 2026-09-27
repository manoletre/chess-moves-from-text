"""Score a trained tagger on a dataset split (span-level precision / recall / F1).

    python scripts/evaluate.py                       # model in chess-move-tagger/, data/test.jsonl
    python scripts/evaluate.py --data data/val.jsonl --show-errors
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chess_tagger.model import SPAN_TYPES, ChessTagger  # noqa: E402


def f1_scores(gold: list, predicted: list) -> tuple[float, float, float]:
    true_positives = sum(len(set(g) & set(p)) for g, p in zip(gold, predicted))
    n_predicted = sum(len(p) for p in predicted)
    n_gold = sum(len(g) for g in gold)
    precision = true_positives / n_predicted if n_predicted else 0.0
    recall = true_positives / n_gold if n_gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def score(examples: list, predicted: list) -> dict:
    gold = [[tuple(s) for s in e["spans"]] for e in examples]
    pred = [[tuple(s) for s in p] for p in predicted]
    precision, recall, f1 = f1_scores(gold, pred)
    metrics = {"examples": len(examples), "precision": precision, "recall": recall, "f1": f1,
               "detection_f1": f1_scores([[s[:2] for s in g] for g in gold], [[s[:2] for s in p] for p in pred])[2]}
    for kind in SPAN_TYPES:
        only = lambda spans: [[s for s in example if s[2] == kind] for example in spans]
        metrics[f"f1_{kind}"] = f1_scores(only(gold), only(pred))[2]
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=ROOT / "chess-move-tagger", type=Path)
    parser.add_argument("--data", default=ROOT / "data" / "test.jsonl", type=Path)
    parser.add_argument("--show-errors", action="store_true")
    args = parser.parse_args()

    examples = [json.loads(line) for line in args.data.read_text().splitlines()]
    tagger = ChessTagger(args.model)
    started = time.perf_counter()
    predicted = tagger.predict_spans([e["text"] for e in examples])
    elapsed = time.perf_counter() - started

    groups = {"all": list(range(len(examples)))}
    for i, example in enumerate(examples):
        groups.setdefault(example.get("source", "").split(":")[0] or "unknown", []).append(i)
    results = {name: score([examples[i] for i in idx], [predicted[i] for i in idx]) for name, idx in groups.items()}
    print(json.dumps(results, indent=2))
    print(f"{len(examples)} examples in {elapsed:.1f}s on {tagger.device}", file=sys.stderr)

    if args.show_errors:
        for example, spans in zip(examples, predicted):
            if [tuple(s) for s in spans] != [tuple(s) for s in example["spans"]]:
                print("expected:", example["marked"].replace("\n", "⏎"), file=sys.stderr)
                print("got:     ", [f"{example['text'][s:e]}:{k}" for s, e, k in spans], "\n", file=sys.stderr)


if __name__ == "__main__":
    main()
