"""Score a trained v3 tagger on a dataset split, per group of texts.

    python scripts/evaluate.py                       # model in chess-move-tagger/, data/test.jsonl
    python scripts/evaluate.py --data data/val.jsonl --show-errors 20
    python scripts/evaluate.py --verify              # also: does verify.py reach the same verdicts
                                                     # on predicted labels as on gold labels?
    python scripts/evaluate.py --onnx web/model      # score the 4-bit browser model instead

Spans count only if start, end and type are exactly right. line/stance/frame
accuracy is measured on the spans the model found exactly.
"""

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chess_tagger.markup import render  # noqa: E402
from chess_tagger.metrics import score, summary  # noqa: E402
from chess_tagger.model import ChessTagger, decode  # noqa: E402
from chess_tagger.verify import verify  # noqa: E402


def verdict_agreement(examples, predicted):
    """For texts with a FEN: compare verify() verdicts on gold vs predicted labels, fact by fact."""
    agree, total, by_kind = 0, 0, Counter()
    chosen_same = texts = unverif_gold = unverif_pred = n_pred = 0
    for e, p in zip(examples, predicted):
        if not e.get("fen") or e.get("claims") is None:
            continue
        g, q = verify(e["fen"], e["text"], e), verify(e["fen"], e["text"], p)
        texts += 1
        chosen_same += g["chosen"] == q["chosen"]
        unverif_gold += sum(f["verdict"] is None for f in g["facts"])
        unverif_pred += sum(f["verdict"] is None for f in q["facts"])
        n_pred += len(q["facts"])
        gold = {(f["kind"], f["start"], f["end"]): f["verdict"] for f in g["facts"]}
        pred = {(f["kind"], f["start"], f["end"]): f["verdict"] for f in q["facts"]}
        for key, verdict in gold.items():
            total += 1
            same = pred.get(key, "missing") == verdict
            agree += same
            by_kind[(key[0], same)] += 1
    kinds = sorted({k for k, _ in by_kind})
    return {"facts": total, "same_verdict": agree / total if total else 0.0,
            "same_chosen_move": chosen_same / texts if texts else 0.0,
            "unverifiable_share_gold": unverif_gold / total if total else 0.0,
            "unverifiable_share_pred": unverif_pred / n_pred if n_pred else 0.0,
            "by_kind": {k: by_kind[(k, True)] / max(1, by_kind[(k, True)] + by_kind[(k, False)]) for k in kinds}}


def predict_onnx(folder: Path, tagger: ChessTagger, examples: list) -> list:
    """Same as tagger.predict, but with the browser's 4-bit ONNX model (one text at a time, no padding)."""
    import numpy as np
    import onnxruntime as ort
    import torch

    config = json.loads((folder / "tagger_config.json").read_text())
    session = ort.InferenceSession(b"".join((folder / part).read_bytes() for part in config["onnx_parts"]))
    out = []
    for e in examples:
        enc = tagger.tokenizer(e["text"], return_offsets_mapping=True, truncation=True, max_length=tagger.max_length)
        logits = session.run(None, {"input_ids": np.array([enc["input_ids"]], dtype=np.int64)})[0][0]
        out.append(decode(e["text"], enc["offset_mapping"], torch.from_numpy(logits)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=ROOT / "chess-move-tagger", type=Path)
    parser.add_argument("--data", default=ROOT / "data" / "test.jsonl", type=Path)
    parser.add_argument("--show-errors", type=int, default=0, metavar="N")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--out", type=Path, help="write the metrics as JSON here")
    parser.add_argument("--onnx", type=Path, help="a web/model folder: score the 4-bit ONNX model")
    args = parser.parse_args()

    examples = [json.loads(line) for line in args.data.read_text().splitlines()]
    tagger = ChessTagger(args.model)
    started = time.perf_counter()
    predicted = predict_onnx(args.onnx, tagger, examples) if args.onnx else tagger.predict([e["text"] for e in examples])
    elapsed = time.perf_counter() - started

    groups = {}
    for i, e in enumerate(examples):
        groups.setdefault(e["group"], []).append(i)
        if e["group"] == "commentary":
            groups.setdefault("commentary:" + e["source"].split(":")[0], []).append(i)
        if e["group"] == "synthetic":
            groups.setdefault(e["source"], []).append(i)
    results = {}
    for name, idx in sorted(groups.items()):
        results[name] = score([examples[i] for i in idx], [predicted[i] for i in idx])
        print(f"{name:32s} {len(idx):4d}  {summary(results[name])}")
    if args.verify:
        results["verify_agreement"] = verdict_agreement(examples, predicted)
        v = results["verify_agreement"]
        print(f"verify.py, same verdict on predicted vs gold labels: {v['same_verdict']:.3f} over {v['facts']} facts; "
              f"same chosen move {v['same_chosen_move']:.3f}; unverifiable share gold {v['unverifiable_share_gold']:.3f} "
              f"/ predicted {v['unverifiable_share_pred']:.3f}")
        print("   " + ", ".join(f"{k} {a:.2f}" for k, a in v["by_kind"].items()))
    print(f"{len(examples)} examples in {elapsed:.1f}s on {tagger.device}", file=sys.stderr)
    if args.out:
        args.out.write_text(json.dumps(results, indent=2))

    shown = 0
    for e, p in zip(examples, predicted):
        if shown >= args.show_errors:
            break
        gold_m = [tuple(m[:3]) for m in e["mentions"]]
        gold_c = [tuple(c[:3]) for c in e["claims"] or []]
        if sorted(gold_m) != sorted(tuple(m[:3]) for m in p["mentions"]) or \
                (e["claims"] is not None and sorted(gold_c) != sorted(tuple(c[:3]) for c in p["claims"])):
            shown += 1
            print("\nexpected:", e["marked"].replace("\n", "⏎"))
            try:
                print("got:     ", render(e["text"], p["mentions"], p["claims"]).replace("\n", "⏎"))
            except ValueError:
                print("got:     ", p)


if __name__ == "__main__":
    main()
