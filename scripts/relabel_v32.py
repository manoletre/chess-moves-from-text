"""v3.2 consistency pass: label game-ending wording as `mate` and evaluations as `eval` in existing labels.

Only adds a claim where the text matches one of the patterns below and no existing span overlaps it;
the frame follows the text-order rule (now before the text's first move, after from then on).
See data/reflections/LABELING.md section 7.

    python scripts/relabel_v32.py --dry-run     # counts + samples
    python scripts/relabel_v32.py
"""

import argparse
import glob
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from chess_tagger.markup import parse, render  # noqa: E402

B, E = r"(?<![\w-])", r"(?![\w-])"
MATE = [
    r"finish(?:es|ed)? (?:it|the game|things off|him off)", r"it'?s over", r"game over", r"the king is (?:dead|done|finished)",
    r"(?:and )?the king falls", r"resign(?:s|ed|ation)?", r"forces? resignation", r"lights out", r"is curtains",
    r"no (?:defen[cs]e|reply|answer) (?:exists|is possible|helps)", r"there is no (?:defen[cs]e|answer|reply)",
    r"can'?t (?:parry|stop) (?:it|the mate|that)", r"cannot (?:parry|stop) (?:it|the mate|that)",
    r"there is nothing to be done", r"nothing can be done", r"seals? (?:the game|the king'?s fate|his fate|it)",
]
EVAL = [
    r"(?:is |are |was |would be )?(?:not )?decisive(?:ly)?", r"(?:winning|strongest|best|only|correct|right|killer) "
    r"(?:move|idea|continuation|line|try|blow|choice|option|plan|shot|way|answer|winning move)",
    r"(?:is )?unstoppable", r"(?:is )?crushing", r"(?:is )?lethal", r"(?:is )?devastating", r"forced win", r"wins by force",
    r"(?:is |are )(?:completely |totally |simply )?(?:lost|busted|hopeless)",
    r"(?:is |would be )(?:a |an )(?:mistake|blunder|error)", r"falls short", r"(?:is )?refuted",
]
PATTERNS = [(re.compile(B + p + E, re.I), "MATE") for p in MATE] + [(re.compile(B + p + E, re.I), "EVAL") for p in EVAL]


def relabel(marked: str):
    parsed = parse(marked)
    text, mentions, claims = parsed["text"], parsed["mentions"], parsed["claims"]
    first_move = min((m[0] for m in mentions if m[2] == "MOVE"), default=None)
    taken = [(m[0], m[1]) for m in mentions] + [(c[0], c[1]) for c in claims]
    added = []
    for pattern, kind in PATTERNS:
        for m in pattern.finditer(text):
            s, e = m.start(), m.end()
            if text[s:e].lower().startswith("the ") and kind == "EVAL":
                s += 4
            if any(a < e and b > s for a, b in taken):
                continue
            if kind == "EVAL" and re.search(r"(lost|busted|hopeless)", text[s:e], re.I) and \
                    not re.search(r"\b(white|black|he|it|they|side|position)\b[\w\s']{0,6}$", text[max(0, s - 20):s], re.I):
                continue  # "is lost" only about a side or position ("the knight is lost" is about material)
            frame = "NOW" if first_move is None or s < first_move else "AFTER"
            claims.append([s, e, kind, frame])
            taken.append((s, e))
            added.append((kind, text[s:e]))
    if not added:
        return marked, []
    claims.sort()
    new = render(text, mentions, claims)
    assert parse(new)["text"] == text
    return new, added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    files = sorted(glob.glob(str(ROOT / "data/reflections/labels/*.jsonl"))) + \
        [f for f in sorted(glob.glob(str(ROOT / "data/reflections/synthetic/*.jsonl"))) if "templated" not in f]
    counts, samples = Counter(), []
    for f in files:
        rows = [json.loads(l) for l in Path(f).read_text().splitlines()]
        changed = False
        for r in rows:
            new, added = relabel(r["marked"])
            if added:
                changed = True
                r["marked"] = new
                for kind, t in added:
                    counts[(kind, t.lower())] += 1
                    samples.append((Path(f).name, r["id"], kind, t))
        if changed and not args.dry_run:
            Path(f).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(sum(counts.values()), "claims added")
    for (kind, t), n in counts.most_common(40):
        print(f"  {n:4d} {kind} {t}")
    random.Random(0).shuffle(samples)


if __name__ == "__main__":
    main()
