"""Parity check: does web/verify.js (the browser port, run with node) reach the same verdicts as
chess_tagger/verify.py on every gold-labelled text with a FEN?

    python scripts/check_verify_js.py            # prints the counts and the first mismatches
    python scripts/check_verify_js.py -v         # every mismatch

Texts: data/reflections/labels/*.jsonl (fen and played from data/reflections/raw.jsonl), every
data/reflections/synthetic/*.jsonl row with a fen, and data/probe/refute.jsonl. The solution move
(`best`, for EVAL claims) is the row's `played` or `chosen`; rows with neither use the move verify.py
chooses, so their evaluations are checked too. Compared per text: every fact's verdict (facts matched
by kind and span), which facts are refutations (and of what), the chosen move, false_refutations and
the contradiction (none or not, and its move). Needs node on the PATH.
"""

import argparse
import glob
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chess_tagger.markup import parse  # noqa: E402
from chess_tagger.verify import verify  # noqa: E402

NODE = """
import { createInterface } from "node:readline";
const { verify } = await import(process.argv[1]);
const uci = (mv) => (mv?.move ? mv.move.from + mv.move.to + (mv.move.promotion ?? "") : null);
for await (const line of createInterface({ input: process.stdin })) {
  const r = JSON.parse(line);
  let out;
  try {
    const rep = verify(r.fen, r.text, { mentions: r.mentions, claims: r.claims }, r.best);
    out = { facts: rep.facts.map((f) => [f.kind, f.start, f.end, f.verdict, f.refutes ?? null]),
      chosen: uci(rep.chosen), false_refutations: rep.false_refutations,
      contradiction: rep.contradiction && rep.contradiction.move };
  } catch (e) { out = { error: String(e.stack || e) }; }
  process.stdout.write(JSON.stringify(out) + "\\n");
}
"""


def load_rows():
    raw = {r["id"]: r for r in map(json.loads, open(ROOT / "data/reflections/raw.jsonl"))}
    rows = []
    for path in sorted(glob.glob(str(ROOT / "data/reflections/labels/*.jsonl"))):
        for r in map(json.loads, open(path)):
            src = raw.get(r["id"])
            if src and src.get("fen"):
                rows.append({"id": r["id"], "marked": r["marked"], "fen": src["fen"], "best": src.get("played")})
    paths = sorted(glob.glob(str(ROOT / "data/reflections/synthetic/*.jsonl"))) + [str(ROOT / "data/probe/refute.jsonl")]
    for path in paths:
        for i, r in enumerate(map(json.loads, open(path))):
            if r.get("fen"):
                rows.append({"id": r.get("id") or f"{Path(path).stem}:{i}", "marked": r["marked"], "fen": r["fen"],
                             "best": r.get("played") or r.get("chosen")})
    return rows


def python_side(row):
    p = parse(row["marked"])
    best = row["best"] or verify(row["fen"], p["text"], p)["chosen"]
    rep = verify(row["fen"], p["text"], p, best=best)
    out = {"facts": [[f["kind"], f["start"], f["end"], f["verdict"], f.get("refutes")] for f in rep["facts"]],
           "chosen": rep["chosen"], "false_refutations": rep["false_refutations"],
           "contradiction": rep["contradiction"] and rep["contradiction"]["move"]}
    return {"fen": row["fen"], "text": p["text"], "mentions": p["mentions"], "claims": p["claims"], "best": best}, out


def by_span(facts):
    grouped = defaultdict(list)
    for kind, start, end, verdict, refutes in facts:
        grouped[(kind, start, end)].append((verdict, refutes))
    return {k: sorted(v, key=repr) for k, v in grouped.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    rows = load_rows()
    inputs, expected = zip(*(python_side(r) for r in rows))
    js = subprocess.run(["node", "--input-type=module", "-e", NODE, (ROOT / "web/verify.js").as_uri()],
                        input="".join(json.dumps(i) + "\n" for i in inputs), capture_output=True, text=True, check=True)
    got = [json.loads(line) for line in js.stdout.splitlines()]
    assert len(got) == len(rows), (len(got), len(rows), js.stderr[-2000:])

    n_facts = n_ref = 0
    problems = []
    for row, inp, py, jsr in zip(rows, inputs, expected, got):
        if "error" in jsr:
            problems.append((row["id"], "js error", jsr["error"].splitlines()[0], ""))
            continue
        n_facts += len(py["facts"])
        n_ref += sum(f[4] is not None for f in py["facts"])
        a, b = by_span(py["facts"]), by_span(jsr["facts"])
        for key in sorted(set(a) | set(b), key=lambda k: (k[1], k[2], k[0])):
            if a.get(key) != b.get(key):
                problems.append((row["id"], f"fact {key[0]} {inp['text'][key[1]:key[2]]!r}", a.get(key), b.get(key)))
        for field in ("chosen", "false_refutations", "contradiction"):
            if py[field] != jsr[field]:
                problems.append((row["id"], field, py[field], jsr[field]))

    bad_texts = len({p[0] for p in problems})
    print(f"{len(rows)} texts, {n_facts} facts ({n_ref} refutations) compared: "
          f"{len(problems)} mismatches in {bad_texts} texts")
    for p in problems if args.verbose else problems[:30]:
        print(f"  {p[0]}  {p[1]}\n      py: {p[2]}\n      js: {p[3]}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
