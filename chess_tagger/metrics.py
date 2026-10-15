"""Scores for the v3 tagger: span F1 for mentions and claims, accuracy for the attributes.

Attribute accuracy (line, stance, frame) is measured on the spans the model found
exactly right, so it isn't double-counting span errors.
"""


def f1(gold: list, predicted: list) -> dict:
    tp = sum(len(set(g) & set(p)) for g, p in zip(gold, predicted))
    n_pred, n_gold = sum(len(set(p)) for p in predicted), sum(len(set(g)) for g in gold)
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gold if n_gold else 0.0
    return {"p": precision, "r": recall, "f1": 2 * precision * recall / (precision + recall) if tp else 0.0,
            "n": n_gold}


def score(examples: list[dict], predicted: list[dict]) -> dict:
    gm = [[tuple(m[:3]) for m in e["mentions"]] for e in examples]
    pm = [[tuple(m[:3]) for m in p["mentions"]] for p in predicted]
    out = {"examples": len(examples), "mention": f1(gm, pm),
           "mention_detection": f1([[m[:2] for m in g] for g in gm], [[m[:2] for m in p] for p in pm])}
    for kind in ("MOVE", "PIECE", "SQUARE", "FILE", "RANK", "DIAGONAL"):
        only = lambda rows: [[m for m in r if m[2] == kind] for r in rows]
        out[f"mention_{kind}"] = f1(only(gm), only(pm))

    full = [i for i, e in enumerate(examples) if e.get("claims") is not None]
    if full:
        gc = [[tuple(c[:3]) for c in examples[i]["claims"]] for i in full]
        pc = [[tuple(c[:3]) for c in predicted[i]["claims"]] for i in full]
        out["claim"] = f1(gc, pc)
        out["claim_detection"] = f1([[c[:2] for c in g] for g in gc], [[c[:2] for c in p] for p in pc])
        for kind in ("CHECK", "MATE", "WINS", "ATTACKS", "FLIGHT", "EVAL"):
            only = lambda rows: [[c for c in r if c[2] == kind] for r in rows]
            out[f"claim_{kind}"] = f1(only(gc), only(pc))

    # attributes on exactly matched spans
    hits = {"line": [0, 0], "stance": [0, 0], "frame": [0, 0]}
    for e, p in zip(examples, predicted):
        pred_m = {tuple(m[:3]): m for m in p["mentions"]}
        pred_c = {tuple(c[:3]): c for c in p["claims"]}
        for m in e["mentions"]:
            q = pred_m.get(tuple(m[:3]))
            if q is None:
                continue
            for name, i in (("line", 3), ("stance", 4), ("frame", 5)):
                if m[i] is not None:
                    hits[name][0] += m[i] == q[i]
                    hits[name][1] += 1
        for c in e.get("claims") or []:
            q = pred_c.get(tuple(c[:3]))
            if q is not None and c[3] is not None:
                hits["frame"][0] += c[3] == q[3]
                hits["frame"][1] += 1
    for name, (right, total) in hits.items():
        out[f"{name}_acc"] = {"acc": right / total if total else 0.0, "n": total}
    return out


def headline(m: dict) -> float:
    """One number to pick the best epoch with."""
    parts = [m["mention"]["f1"], m.get("claim", {}).get("f1", 0.0)]
    parts += [m[f"{k}_acc"]["acc"] for k in ("line", "stance", "frame") if m[f"{k}_acc"]["n"]]
    return sum(parts) / len(parts)


def summary(m: dict) -> str:
    s = f"mention F1 {m['mention']['f1']:.3f}"
    if "claim" in m:
        s += f" | claim F1 {m['claim']['f1']:.3f}"
    for k in ("line", "stance", "frame"):
        if m[f"{k}_acc"]["n"]:
            s += f" | {k} {m[f'{k}_acc']['acc']:.3f}"
    return s
