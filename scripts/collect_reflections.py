"""Collect real reflections written by the chessLLM puzzle model (LFM2.5-1.2B, SFT + GRPO).

Each reflection is the text the model writes after get_legal_moves and before
make_move. We keep the position it was written for (FEN) and the move it then
played, so the labels can be checked with python-chess.

    python scripts/collect_reflections.py ~/Dev/chessLLM/results

Writes data/reflections/raw.jsonl: {id, source, puzzle_id, fen, played, text}.
"""

import argparse
import glob
import json
import random
from pathlib import Path

import chess

OUT = Path(__file__).parent.parent / "data" / "reflections" / "raw.jsonl"
TOOL_CALL = "<|tool_call_start|>"

# How many reflections to take from each source (cur-v2 = the curriculum GRPO run,
# stage0 = SFT v2 before RL; tsweep = SFT v2 / stage checkpoints sampled at different temperatures).
QUOTA = {"cur-v2-stage0": 360, "cur-v2-stage1": 70, "cur-v2-stage2": 70, "cur-v2-stage3": 70,
         "cur-v2-stage4": 70, "tsweep": 160}


def bench_rows(path: str, source: str):
    for line in open(path):
        record = json.loads(line)
        for i, step in enumerate(record["steps"]):
            if step.get("reflection"):
                yield {"source": source, "puzzle_id": record["puzzle_id"], "fen": step["fen"],
                       "played": step.get("model_move"), "text": step["reflection"]}


def sweep_rows(path: str):
    name = Path(path).parent.name + "/" + Path(path).stem
    for line in open(path):
        record = json.loads(line)
        for sample in record["samples"]:
            text = sample["text"].split(TOOL_CALL)[0]
            played = None
            if TOOL_CALL in sample["text"] and "make_move(move=" in sample["text"]:
                played = sample["text"].split("make_move(move=")[1][1:6].strip("\"')")
            yield {"source": f"tsweep:{name}", "puzzle_id": record["puzzle_id"], "fen": record["fen"],
                   "played": played, "text": text}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", help="chessLLM results/ directory")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    pools = {}
    for stage in range(5):
        source = f"cur-v2-stage{stage}"
        pools[source] = list(bench_rows(f"{args.results}/cur-v2-traces/{source}-bench.jsonl", source))
    pools["tsweep"] = [row for path in sorted(glob.glob(f"{args.results}/tsweep[34]/*.jsonl"))
                       for row in sweep_rows(path)]

    rows, seen = [], set()
    for source, pool in pools.items():
        rng.shuffle(pool)
        taken = 0
        for row in pool:
            text = row["text"].strip()
            if taken == QUOTA[source] or not text:
                continue
            if text in seen or len(text.split()) < 12:
                continue
            chess.Board(row["fen"])  # must parse
            seen.add(text)
            rows.append({**row, "text": text})
            taken += 1
        print(f"{source}: {taken} of {len(pool)}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for i, row in enumerate(rows):
            f.write(json.dumps({"id": f"r{i:04d}", **row}) + "\n")
    print(f"{len(rows)} reflections -> {OUT}")


if __name__ == "__main__":
    main()
