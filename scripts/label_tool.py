"""Helpers for hand-labelling reflections in the v3 markup (see data/reflections/LABELING.md).

    python scripts/label_tool.py batches            # raw.jsonl -> split + batches/bNN.jsonl to label
    python scripts/label_tool.py check labels/b03.jsonl   # validate one labelled batch
    python scripts/label_tool.py check-all --frames # validate every labelled batch, frames required
    python scripts/label_tool.py add-frames labels/b03.jsonl  # fill missing frames by the text-order rule
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).parent.parent))
from chess_tagger.markup import check, parse, render  # noqa: E402

DIR = Path(__file__).parent.parent / "data" / "reflections"
BATCH_SIZE = 50
SPLIT = {"test": 0.2, "val": 0.075}  # by puzzle, so a test puzzle is never seen in training


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def batches():
    rows = load(DIR / "raw.jsonl")
    puzzles = sorted({r["puzzle_id"] for r in rows})
    random.Random(0).shuffle(puzzles)
    n_test, n_val = int(SPLIT["test"] * len(puzzles)), int(SPLIT["val"] * len(puzzles))
    split_of = {p: "test" for p in puzzles[:n_test]}
    split_of.update({p: "val" for p in puzzles[n_test:n_test + n_val]})
    for row in rows:
        row["split"] = split_of.get(row["puzzle_id"], "train")
        board = chess.Board(row["fen"])
        row["side_to_move"] = "White" if board.turn else "Black"
        try:
            row["played_san"] = board.san(chess.Move.from_uci(row["played"]))
        except (ValueError, TypeError, AssertionError):
            row["played_san"] = None
    print(Counter(r["split"] for r in rows))
    (DIR / "raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = DIR / "batches"
    out.mkdir(exist_ok=True)
    order = rows[:]
    random.Random(1).shuffle(order)
    for i in range(0, len(order), BATCH_SIZE):
        keep = ["id", "fen", "side_to_move", "played_san", "text"]
        (out / f"b{i // BATCH_SIZE:02d}.jsonl").write_text(
            "".join(json.dumps({k: r[k] for k in keep}) + "\n" for r in order[i:i + BATCH_SIZE]))
    print(f"{(len(order) + BATCH_SIZE - 1) // BATCH_SIZE} batches in {out}")


def check_file(path, frames: bool = False) -> int:
    raw = {r["id"]: r for r in load(DIR / "raw.jsonl")}
    synthetic = {r["id"]: r for p in (DIR / "synthetic").glob("*.jsonl") for r in load(p)} \
        if (DIR / "synthetic").exists() else {}
    n_bad, stats = 0, Counter()
    for row in load(path):
        original = raw[row["id"]]["text"] if row["id"] in raw else None
        problems = check(row["marked"], original, frames=frames)
        if row["id"] not in raw and row["id"] not in synthetic and not row["id"].startswith(("s", "p", "t")):
            problems.append("unknown id")
        if problems:
            n_bad += 1
            print(f"{path.name if hasattr(path, 'name') else path} {row['id']}: " + "; ".join(problems))
            continue
        parsed = parse(row["marked"])
        stats.update(m[2] for m in parsed["mentions"])
        stats.update("claim:" + c[2] for c in parsed["claims"])
    print(f"{path}: {n_bad} problems | " + ", ".join(f"{k} {v}" for k, v in sorted(stats.items())))
    return n_bad


def add_frames(path) -> None:
    """Fill in missing frames with the rule the labellers converged on: a piece or claim before
    the text's first move is `now`, after it `after` (frames already present are kept)."""
    rows = load(path)
    for row in rows:
        parsed = parse(row["marked"])
        first_move = min((m[0] for m in parsed["mentions"] if m[2] == "MOVE"), default=None)
        frame = lambda start: "NOW" if first_move is None or start < first_move else "AFTER"
        for m in parsed["mentions"]:
            if m[2] == "PIECE" and m[5] is None:
                m[5] = frame(m[0])
        for c in parsed["claims"]:
            if c[3] is None:
                c[3] = frame(c[0])
        marked = render(parsed["text"], parsed["mentions"], parsed["claims"])
        assert parse(marked)["text"] == parsed["text"]
        row["marked"] = marked
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "batches":
        batches()
    elif command == "check":  # add --frames to also require a frame on every piece and claim
        sys.exit(1 if check_file(Path(sys.argv[2]), "--frames" in sys.argv) else 0)
    elif command == "add-frames":
        add_frames(Path(sys.argv[2]))
    elif command == "check-all":
        files = sorted((DIR / "labels").glob("*.jsonl")) + sorted((DIR / "synthetic").glob("*.jsonl"))
        sys.exit(1 if sum(check_file(p, "--frames" in sys.argv) for p in files) else 0)
