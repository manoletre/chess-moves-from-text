"""Score a tagger on generated reflections (scripts/generated_labels.py). The text is correct by construction, so:
- span scores come from chess_tagger.metrics against the generator's labels;
- every False verdict from verify.py is a false alarm: --gold shows the ones the labels themselves cause (the checker's
  share), the model's own column shows what the tagger adds.

    python scripts/eval_generated.py data/generated/test_natural.jsonl --model chess-move-tagger --out baseline.json
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from chess_tagger.markup import render  # noqa: E402
from chess_tagger.metrics import score, summary  # noqa: E402
from chess_tagger.verify import verify  # noqa: E402

ALL_TYPES = ("MOVE", "PIECE", "SQUARE", "REGION", "FILE", "RANK", "DIAGONAL")
CLAIMS = ("CHECK", "MATE", "WINS", "ATTACKS", "FLIGHT", "EVAL")


def replay(examples, tagged) -> dict:
    """What the reward would see: same moves replayed, chosen move, false alarms, phantom contradictions."""
    out = Counter()
    kinds = Counter()
    for e, t in zip(examples, tagged):
        r = verify(e["fen"], e["text"], t, played=e["played"], best=None)
        out["texts"] += 1
        out["chosen_right"] += r["chosen"] == e["played"]
        out["contradiction"] += r["contradiction"] is not None
        out["any_false"] += any(f["verdict"] is False for f in r["facts"])
        for f in r["facts"]:
            out["facts"] += 1
            out["false"] += f["verdict"] is False
            out["none"] += f["verdict"] is None
            if f["verdict"] is False:
                kinds[f["kind"]] += 1
    n = out["texts"]
    return {"texts": n, "chosen_right": out["chosen_right"] / n, "phantom_contradiction": out["contradiction"] / n,
            "texts_with_false_fact": out["any_false"] / n, "facts": out["facts"], "false_share": out["false"] / out["facts"],
            "unverifiable_share": out["none"] / out["facts"], "false_by_kind": dict(kinds.most_common())}


def uci_recall(examples, tagged) -> dict:
    """Moves resolved to the right UCI at the right place in the line, whatever the span's exact edges."""
    hit = total = 0
    for e, t in zip(examples, tagged):
        gold = verify(e["fen"], e["text"], e)
        pred = verify(e["fen"], e["text"], t)
        g = [f["detail"] for f in gold["facts"] if f["kind"] == "legal_move" and f["verdict"] is True]
        p = Counter(f["detail"] for f in pred["facts"] if f["kind"] == "legal_move" and f["verdict"] is True)
        for d in g:
            total += 1
            if p[d] > 0:
                hit += 1
                p[d] -= 1
    return {"recall": hit / total if total else 0.0, "n": total}


def failures(examples, tagged, top=12) -> dict:
    """The most common wrong spans, with pieces, squares and SANs masked so the same mistake is counted once."""
    def mask(s):
        s = re.sub(r"\b(?:pawn|knight|bishop|rook|queen|king)\b", "P", s)
        s = re.sub(r"(?<![\w-])(?:O-O(?:-O)?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)(?![\w])", "M", s)
        return re.sub(r"[+-]?\d+(?:\.\d+)?", "N", s)
    fp_m, fn_m, fp_c, fn_c, stance, line = Counter(), Counter(), Counter(), Counter(), Counter(), Counter()
    for e, t in zip(examples, tagged):
        text = e["text"]
        gm, pm = {tuple(m[:3]): m for m in e["mentions"]}, {tuple(m[:3]): m for m in t["mentions"]}
        gc, pc = {tuple(c[:3]) for c in e["claims"]}, {tuple(c[:3]) for c in t["claims"]}
        for k in pm.keys() - gm.keys():
            fp_m[f"{k[2]}: {mask(text[k[0]:k[1]])}"] += 1
        for k in gm.keys() - pm.keys():
            fn_m[f"{k[2]}: {mask(text[k[0]:k[1]])}"] += 1
        for k in pc - gc:
            fp_c[f"{k[2]}: {mask(text[k[0]:k[1]])}"] += 1
        for k in gc - pc:
            fn_c[f"{k[2]}: {mask(text[k[0]:k[1]])}"] += 1
        for k in gm.keys() & pm.keys():
            if k[2] == "MOVE":
                if gm[k][4] != pm[k][4]:
                    stance[f"{gm[k][4]} read as {pm[k][4]}"] += 1
                if gm[k][3] != pm[k][3]:
                    line[f"{gm[k][3]} read as {pm[k][3]}"] += 1
    return {"spurious_mentions": fp_m.most_common(top), "missed_mentions": fn_m.most_common(top),
            "spurious_claims": fp_c.most_common(top), "missed_claims": fn_c.most_common(top),
            "stance_errors": stance.most_common(), "line_errors": line.most_common()}


def report(examples, tagged) -> dict:
    m = score(examples, tagged)
    out = {"spans": {k: {"f1": round(v["f1"], 3), "p": round(v["p"], 3), "r": round(v["r"], 3), "n": v["n"]}
                     for k, v in m.items() if isinstance(v, dict) and "f1" in v},
           "attributes": {k: {"acc": round(m[k]["acc"], 3), "n": m[k]["n"]} for k in ("line_acc", "stance_acc", "frame_acc")},
           "uci_recall": uci_recall(examples, tagged), "replay": replay(examples, tagged), "failures": failures(examples, tagged)}
    out["summary"] = summary(m)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("--model", type=Path)
    ap.add_argument("--gold", action="store_true", help="run verify on the gold labels only (no model)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--show", type=int, default=0)
    a = ap.parse_args()
    examples = [json.loads(line) for line in a.data.read_text().splitlines()]
    if a.gold:
        tagged = examples
    else:
        from chess_tagger.model import ChessTagger
        tagged = ChessTagger(a.model).predict([e["text"] for e in examples])
    result = report(examples, tagged) if not a.gold else {"replay": replay(examples, tagged)}
    print(json.dumps(result, indent=1))
    if a.out:
        a.out.write_text(json.dumps({**result, "tagged": tagged if not a.gold else None}))
    for e, t in list(zip(examples, tagged))[:a.show]:
        print("\nexpected:", e["marked"])
        print("got:     ", render(e["text"], t["mentions"], [c for c in t["claims"]]))


if __name__ == "__main__":
    main()
