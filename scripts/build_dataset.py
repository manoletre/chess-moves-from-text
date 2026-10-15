"""Build the v3 dataset: data/{train,val,test}.jsonl.

Three sources:
  reflections  real reflections from the chessLLM puzzle model, hand-labelled
               (data/reflections/raw.jsonl + labels/*.jsonl; split by puzzle)
  synthetic    hand-written reflections in other styles (data/reflections/synthetic/*.jsonl)
  commentary   the v1/v2 chess-tutor data (data/commentary/), converted: START -> MOVE new,
               CONT -> MOVE next, pieces added with a regex. These texts have no stance or
               claim labels (nor frames, nor alt lines), so those heads are masked out for them
               during training.

Every line: {"text", "mentions": [[s, e, type, line, stance, frame]], "claims": [[s, e, type, frame]] | null,
             "labelled": "full" | "mentions", "marked", "group", "source", "fen"?, "played"?}

    python scripts/build_dataset.py
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from chess_tagger.markup import check, parse, render  # noqa: E402

FRAMES_REQUIRED = True

DATA = ROOT / "data"
REFL = DATA / "reflections"
PIECE = re.compile(r"\b(?:(?:White's|Black's|white|black|White|Black)\s+)?"
                   r"(?:king|queen|rook|bishop|knight|pawn)s?\b(?!'s Indian|'s Gambit|side)", re.I)
COMMENTARY_VAL = 75  # of the 225 v2 validation texts, so reflections dominate model selection


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def from_markup(marked, **extra):
    parsed = parse(marked)
    return {"text": parsed["text"], "mentions": parsed["mentions"], "claims": parsed["claims"],
            "labelled": "full", "marked": marked, **extra}


def reflections():
    raw = {r["id"]: r for r in load(REFL / "raw.jsonl")}
    out = {"train": [], "val": [], "test": []}
    for path in sorted((REFL / "labels").glob("*.jsonl")):
        for row in load(path):
            r = raw[row["id"]]
            problems = check(row["marked"], r["text"], frames=FRAMES_REQUIRED)
            if problems:
                raise SystemExit(f"{path.name} {row['id']}: {problems}")
            out[r["split"]].append(from_markup(row["marked"], group="reflection", source=r["source"], id=r["id"],
                                               fen=r["fen"], played=r["played"]))
    return out


def synthetic():
    out = {"train": [], "test": []}
    rows = [row for path in sorted((REFL / "synthetic").glob("*.jsonl")) for row in load(path)]
    seen = Counter()
    for row in rows:
        problems = check(row["marked"], frames=FRAMES_REQUIRED)
        if problems:
            raise SystemExit(f"synthetic {row['id']}: {problems}")
        category = row.get("category", "other")
        seen[category] += 1
        # a quarter of every category is held out; templated text is never tested here (data/probe/ is its test)
        split = "train" if category == "templated" else "test" if seen[category] % 4 == 0 else "train"
        out[split].append(from_markup(row["marked"], group="synthetic", source=f"synthetic:{category}",
                                      id=row["id"], fen=row.get("fen")))
    return out


def convert_commentary(example):
    mentions = []
    for start, end, kind in example["spans"]:
        if kind in ("START", "CONT"):
            mentions.append([start, end, "MOVE", "NEW" if kind == "START" else "NEXT", None, None])
        else:
            mentions.append([start, end, kind, None, None, None])
    taken = [(s, e) for s, e, *_ in mentions]
    for m in PIECE.finditer(example["text"]):
        if not any(m.start() < e and m.end() > s for s, e in taken):
            mentions.append([m.start(), m.end(), "PIECE", None, None, None])
    mentions.sort()
    first = next((m for m in mentions if m[2] == "MOVE"), None)
    if first and first[3] == "NEXT":
        first[3] = "NEW"
    marked = render(example["text"], [[s, e, k, line, stance or "CONSIDERED", None] for s, e, k, line, stance, _ in mentions], [])
    return {"text": example["text"], "mentions": mentions, "claims": None, "labelled": "mentions",
            "marked": marked, "group": "commentary", "source": example.get("source", "")}


def main():
    refl, synth = reflections(), synthetic()
    commentary = {split: [convert_commentary(e) for e in load(DATA / "commentary" / f"{split}.jsonl")]
                  for split in ("train", "val", "test")}
    splits = {
        "train": refl["train"] + synth["train"] + commentary["train"],
        "val": refl["val"] + commentary["val"][:COMMENTARY_VAL],
        "test": refl["test"] + synth["test"] + commentary["test"],
    }
    for name, rows in splits.items():
        (DATA / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"{name}: {len(rows)}  " + ", ".join(f"{k} {v}" for k, v in Counter(r["group"] for r in rows).items()))


if __name__ == "__main__":
    main()
