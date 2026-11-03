"""Is a generated SFT set any good? Reports variety, truth (re-checked on the board), answer position, leakage,
coverage and length. Usage: reflect_metrics.py rows.jsonl [limit] [old_templates.py]"""
import importlib.util
import json
import math
import random
import re
import sys
import time
from collections import Counter

import chess
import numpy as np

import reflect_templates as T

SAN = re.compile(r"(?<![\w-])(?:O-O(?:-O)?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)(?![\w])")
PIECE = re.compile(r"\b(?:pawn|knight|bishop|rook|queen|king)s?\b")
NUM = re.compile(r"[+-]?\d+(?:\.\d+)?")


def mask(t):
    return NUM.sub("N", PIECE.sub("P", SAN.sub("M", t))).lower()


def sentences(t):
    return [s.strip() for s in re.split(r"(?<=[.?!])\s+", t) if s.strip()]


def grams(t, n):
    w = mask(t).split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def variety(texts, rng):
    sents = [mask(s) for t in texts for s in sentences(t)]
    frames, starts = Counter(sents), Counter(" ".join(s.split()[:3]) for s in sents)
    five = Counter(g for t in texts for g in grams(t, 5))
    tri = [grams(t, 3) for t in texts]
    pairs = [rng.sample(range(len(tri)), 2) for _ in range(3000)]
    jac = sorted(len(tri[a] & tri[b]) / max(1, len(tri[a] | tri[b])) for a, b in pairs)
    H = -sum(c / len(sents) * math.log(c / len(sents)) for c in starts.values())
    top3 = five.most_common(3)
    n_s = len(sents)
    return {"unique_texts": len(set(texts)) / len(texts), "unique_masked": len({mask(t) for t in texts}) / len(texts),
            "pair_jaccard_median": jac[len(jac) // 2], "pair_jaccard_p99": jac[int(len(jac) * .99)],
            "top3_5grams": [(g, round(c / len(texts), 3)) for g, c in top3],
            "top3_sentences": [(g, round(c / n_s, 4)) for g, c in frames.most_common(3)],
            "top3_sentence_share": sum(c for _, c in frames.most_common(3)) / n_s,
            "sentence_start_entropy": H / math.log(max(2, len(starts))), "distinct_starts": len(starts)}


def truth(rows, out):
    """Replays everything from the row with code separate from the generator. Returns failures per kind and totals."""
    bad, n = Counter(), Counter()
    for r, o in zip(rows, out):
        root = chess.Board(r["fen"])
        n["final"] += 1
        if root.san(chess.Move.from_uci(o["best"])) not in o["final"] or o["best"] != r["multipv"][0]["uci"]:
            bad["final"] += 1
        for c in o["cands"]:
            n["lines"] += 1
            b = root.copy()
            ok = True
            for s, u in zip([c["san"], *c["sans"]], c["cand"]["pv"]):
                if b.san(chess.Move.from_uci(u)) != s or s not in c["core"]:
                    ok = False
                b.push(chess.Move.from_uci(u))
            bad["lines"] += not ok
            if c["material"]:
                n["material"] += 1
                val = lambda col: sum(T.VALUE[p.piece_type] * (1 if p.color == col else -1) for p in b.piece_map().values())
                d = val(root.turn) - sum(T.VALUE[p.piece_type] * (1 if p.color == root.turn else -1)
                                         for p in root.piece_map().values())
                num = int(NUM.findall(c["material"])[0])
                signed = num if re.search(r"\bup\b|gained", c["material"]) else -num
                bad["material"] += signed != d

            mv = chess.Move.from_uci(c["uci"])
            after = root.copy()
            after.push(mv)
            for kind, text in c["facts"]:
                n["fact_" + kind] += 1
                if kind == "mate":
                    ok = after.is_checkmate()
                elif kind == "check":
                    ok = after.is_check()
                elif kind in ("attack", "fork"):
                    found = re.findall(r"the (undefended )?(\w+) on ([a-h][1-8])", text)
                    ok = bool(found)
                    for und, piece, sq in found:
                        s = chess.parse_square(sq)
                        p = after.piece_at(s)
                        ok &= (p is not None and p.color != root.turn and T.NAME[p.piece_type] == piece
                               and s in after.attacks(mv.to_square)
                               and (not und or not after.attackers(after.turn, s)))
                elif kind == "risk":
                    piece, sq = re.search(r"their (\w+) on ([a-h][1-8])", text).groups()
                    s = chess.parse_square(sq)
                    ok = T.NAME[after.piece_type_at(s)] == piece and after.is_legal(chess.Move(s, mv.to_square))
                else:
                    ok = root.is_attacked_by(not root.turn, mv.from_square)
                bad["fact_" + kind] += not ok

            n["verdict"] += 1
            cand, v = c["cand"], c["verdict"].lower()
            if cand["mate"] is not None:
                ok = str(abs(cand["mate"])) in v and (cand["mate"] > 0) == ("for me" in v or "i mate" in v or "i have" in v)
            else:
                pool = next(p for lo, p in T.BANDS if cand["cp"] >= lo)
                ok = any(w.lower() in v for w in pool)
                num = re.search(r"[+-]\d+\.\d", v)
                ok &= not num or num.group() == f"{cand['cp'] / 100:+.1f}"
            bad["verdict"] += not ok
    return {k: f"{bad[k]}/{n[k]}" for k in sorted(n)}


def position(out):
    by_n = {}
    for o in out:
        by_n.setdefault(len(o["order"]), Counter())[o["order"].index(o["best"]) + 1] += 1
    report = {}
    for n, c in sorted(by_n.items()):
        total = sum(c.values())
        exp = total / n
        report[n] = {"share": [round(c[i + 1] / total, 3) for i in range(n)],
                     "chi2": round(sum((c[i + 1] - exp) ** 2 / exp for i in range(n)), 2), "df": n - 1, "n": total}
    return report


def cheap_guessers(out):
    """How often does picking by shape, without reading chess, find the best move? Chance is the mean of 1/n."""
    chance = np.mean([1 / len(o["order"]) for o in out])
    rules = {"first": lambda c: 0, "last": lambda c: len(c) - 1,
             "longest_line": lambda c: max(range(len(c)), key=lambda i: len(c[i]["sans"])),
             "most_words": lambda c: max(range(len(c)), key=lambda i: len(c[i]["core"].split()))}
    res = {"chance": round(float(chance), 3)}
    for k, f in rules.items():
        res[k] = round(float(np.mean([o["cands"][f(o["cands"])]["uci"] == o["best"] for o in out])), 3)
    return res


def text_leak(rows, out):
    """Can a classifier spot the best candidate from its masked wording, with evaluations and facts removed?"""
    def feats(c):
        w = mask(c["core"]).split()
        return set(w) | {" ".join(p) for p in zip(w, w[1:])}
    cut = int(len(out) * .8)
    counts = {True: Counter(), False: Counter()}
    docs = Counter()
    for o in out[:cut]:
        for c in o["cands"]:
            label = c["uci"] == o["best"]
            docs[label] += 1
            counts[label].update(feats(c))
    def score(c):                   # naive Bayes log-odds
        return sum(math.log((counts[True][f] + 1) / (docs[True] + 2)) - math.log((counts[False][f] + 1) / (docs[False] + 2))
                   for f in feats(c) if counts[True][f] + counts[False][f] >= 5)
    def auc(keep):
        pos, neg = [], []
        for r, o in zip(rows[cut:], out[cut:]):
            b = chess.Board(r["fen"])
            for c in o["cands"]:
                if keep(b, chess.Move.from_uci(c["uci"])):
                    (pos if c["uci"] == o["best"] else neg).append(score(c))
        a = np.array(pos + neg)
        order = np.argsort(a)
        rk = np.empty(len(a))
        rk[order] = np.arange(1, len(a) + 1)
        return round(float((rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))), 3)
    return {"all": auc(lambda b, m: True), "captures_only": auc(lambda b, m: b.is_capture(m)),
            "non_captures_only": auc(lambda b, m: not b.is_capture(m))}


def rank(a):
    r = np.empty(len(a))
    r[np.argsort(a)] = np.arange(len(a))
    return r


def coverage(rows, out, texts):
    pieces = Counter(), Counter(), Counter(), Counter(), Counter()
    phase, side, mover, best_band, kind = pieces
    gaps, words, ncand = [], [], []
    for r, o, t in zip(rows, out, texts):
        b = chess.Board(r["fen"])
        k = len(b.piece_map())
        phase["opening 27+" if k >= 27 else "middle 13-26" if k >= 13 else "end <=12"] += 1
        side["white" if b.turn else "black"] += 1
        m = chess.Move.from_uci(o["best"])
        mover[T.NAME[b.piece_type_at(m.from_square)]] += 1
        kind["capture" if b.is_capture(m) else "check" if b.gives_check(m) else "quiet"] += 1
        c = r["multipv"][0]
        best_band["mate for me" if (c["mate"] or 0) > 0 else "mated" if (c["mate"] or 0) < 0 else
                  "winning" if c["cp"] >= 500 else "better" if c["cp"] >= 50 else "equal" if c["cp"] >= -50
                  else "worse" if c["cp"] >= -500 else "lost"] += 1
        gaps.append(c["win"] - r["multipv"][1]["win"])
        words.append(len(t.split()))
        ncand.append(len(o["order"]))
    pct = lambda c: {k: round(v / len(rows), 3) for k, v in c.most_common()}
    w = np.array(words)
    return {"phase": pct(phase), "side": pct(side), "best_piece": pct(mover), "best_kind": pct(kind),
            "best_eval": pct(best_band), "single_good_move (gap>=15)": round(float(np.mean(np.array(gaps) >= 15)), 3),
            "words_p5_50_95": [int(np.percentile(w, p)) for p in (5, 50, 95)],
            "corr_gap_vs_words": round(float(np.corrcoef(rank(np.array(gaps)), rank(w))[0, 1]), 3),
            "corr_gap_vs_candidates": round(float(np.corrcoef(rank(np.array(gaps)), rank(np.array(ncand)))[0, 1]), 3)}


if __name__ == "__main__":
    path = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if len(r["multipv"]) >= 3 and all(c.get("pv") for c in r["multipv"]):
                rows.append(r)
            if len(rows) >= limit:
                break
    rng = random.Random(0)

    if len(sys.argv) > 3:
        spec = importlib.util.spec_from_file_location("old_templates", sys.argv[3])
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        print("BEFORE", json.dumps(variety([old.reflect(r)[0] for r in rows], rng), indent=1))

    t0 = time.perf_counter()
    out = [T.reflect(r) for r in rows]
    print(f"rows {len(rows)}  {len(rows) / (time.perf_counter() - t0):.0f} rows/s")
    texts = [o["text"] for o in out]
    print("VARIETY", json.dumps(variety(texts, rng), indent=1))
    print("TRUTH failures", json.dumps(truth(rows, out), indent=1))
    print("BEST MOVE POSITION by number of candidates", json.dumps(position(out)))
    print("SHAPE GUESSERS", json.dumps(cheap_guessers(out)))
    print("TEXT LEAK AUC", text_leak(rows, out))
    print("COVERAGE", json.dumps(coverage(rows, out, texts), indent=1))
    for level in ("easy", "medium", "hard"):
        w = [len(o["text"].split()) for o in out if o["tier"] == level]
        print(level, f"share {len(w) / len(out):.3f} words p5/50/95", [int(np.percentile(w, p)) for p in (5, 50, 95)])
        print("\n" + rng.choice([o["text"] for o in out if o["tier"] == level]) + "\n")
