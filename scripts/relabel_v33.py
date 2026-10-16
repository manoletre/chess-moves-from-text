"""v3.3 consistency pass: label how a candidate is refuted, regions and pressure, so the checker can verify them.

Adds, where the text matches and no existing span overlaps (see data/reflections/LABELING.md section 9):
- `flight`: a king getting away with no square named ("the king simply <steps away>", "<escapes>",
  "<can slip away>", "<escapes to safety>"): the checker reads it as "the king has a legal move".
- `eval`: a candidate dismissed by its outcome ("the <attack fizzles>", "<dissolving the attack>",
  "<lets White consolidate>", "<without a decisive follow-up>", "<prolonging the game>"): a negative
  evaluation, true unless the candidate is the solution.
- `region` mentions: {center}, {kingside}, {queenside}, {corner} (square sets for the checker).
- `attacks`: pressure wording with a target in its clause and a piece/move before it ("the rook <puts pressure on>
  the {center}", "<bearing down on> {f2}").
The frame follows the text-order rule (now before the text's first move, after from then on).

    python scripts/relabel_v33.py --dry-run     # counts + samples
    python scripts/relabel_v33.py
    python scripts/relabel_v33.py --file some.jsonl --key marked   # any jsonl with markup
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
SIDE = r"(?:White|Black|the opponent|the defender|him|them|the enemy)"
MODAL = r"(?:(?:can|could|will|would|may|might|just|then)\s+)?(?:(?:simply|just|easily|calmly|still|quickly)\s+)?"
ESCAPE_VERB = (r"(?:steps? away|stepping away|steps? aside|sidesteps?|escapes?|escaping|flees?|fleeing|slips? away|slipping away|"
               r"slips? out|walks? away|walks? out|runs? away|gets? away|gets? out|wriggles? out|hides?|evades? (?:it|the check)|"
               r"finds? (?:a )?safe(?:ty| square)|escapes? to safety|step to safety|steps? out of (?:the )?check)")
# a king word before the verb in the same clause: "the king simply steps away", "Black's king escapes"
KING_BEFORE = re.compile(r"\bking\b[^.;:,]{0,30}$", re.I)
SQUARE_AFTER = re.compile(r"^\s*(?:to|on|via|towards?|into)\s+(?:the\s+)?[a-h][1-8]", re.I)
NOUN_BEFORE = re.compile(r"(?:'s|\bthe|\bits|\bhis|\bany|\ban)\s+$", re.I)  # "the king's escape", "an escape"
NOUN_AFTER = re.compile(r"^\s*(?:squares?|routes?|paths?|hatch|options?|plan)\b", re.I)
KING_WORD = re.compile(r"\b(?:(?:White's|Black's|white|black|White|Black|the enemy|enemy)\s+)?king\b(?!'s)", re.I)

FLIGHT = [MODAL + ESCAPE_VERB + r"(?:\s+(?:the mating net|the net|the attack|mate|the checks?|to safety)|"
          r"\s+to(?=\s+the\s+(?:center|centre|kingside|queenside|king-side|queen-side|corner)\b))?"]
EVAL = [
    r"(?:dissolv|defus|disrupt|kill|neutraliz|neutralis|blunt|derail)(?:es|ing|e|s)? the (?:whole )?(?:attack|mating net|threat|pressure|initiative|combination)",
    r"(?:attack|mating net|initiative|pressure|threat|combination|net) (?:simply |quickly |completely |just )?(?:fizzles(?: out)?|dissipates|dissolves|"
    r"collapses|stalls|evaporates|peters out|loses (?:its )?(?:steam|momentum)|runs out of steam|dies down|falls apart|is over|vanishes|is gone)",
    r"(?:no|without (?:a|any)) (?:decisive |immediate |forced |clear |real |winning |further |good |concrete )*"
    r"(?:follow-?up|continuation|compensation|winning path|material advantage|decisive advantage|way through|breakthrough)",
    r"(?:lets?|letting|allows?|allowing|allow|giving|gives|give) " + SIDE + r" (?:time )?(?:to )?(?:consolidate|reorgani[sz]e|regroup|escape|survive|defend|breathe|untangle)",
    r"(?:escap(?:e|es|ing)|surviv(?:e|es|ing)) (?:the mating net|the net|the attack|mate)",
    r"(?:again )?(?:losing|loses|lose) (?:the|all) (?:initiative|momentum|attack)",
    r"prolong(?:s|ing)? the game", r"wast(?:es|ing|e) (?:the|a|an|its) (?:opportunity|tempo|chance|time|move|attack)",
    r"(?:is |are |would be )?(?:also |simply |far |much )?too slow", r"(?:is |was |would be )?not (?:immediately |quite |really )?(?:decisive|winning|enough|sufficient)",
    r"(?:for nothing|achieves nothing|leads nowhere|leads to nothing|goes nowhere)",
]
REGION = re.compile(B + r"(?:center|centre|kingside|queenside|king-side|queen-side|corner)" + E, re.I)
PRESSURE = re.compile(B + r"(?:(?:puts?|putting|keeps?|keeping|adds?|adding|exerts?|exerting|piles? up|piling up|increases?|"
                      r"increasing|applies|applying)\s+(?:\w+\s+)?pressure\s+on|pressur(?:es|ing)|bear(?:s|ing)?\s+down\s+on)" + E, re.I)
# side subject + "and escape(s)" / "and survive(s)" after a reply: "White can recapture and escape"
SIDE_ESCAPE = r"(?:and |to )(?:simply )?(?:escape|escapes|survive|survives)"
PATTERNS = [(re.compile(B + p + E, re.I), "FLIGHT") for p in FLIGHT] + \
           [(re.compile(B + p + E, re.I), "EVAL") for p in EVAL] + [(re.compile(B + SIDE_ESCAPE + E, re.I), "EVAL_SIDE")]


def relabel(marked: str):
    parsed = parse(marked)
    text, mentions, claims = parsed["text"], parsed["mentions"], parsed["claims"]
    first_move = min((m[0] for m in mentions if m[2] == "MOVE"), default=None)
    taken = [(m[0], m[1]) for m in mentions] + [(c[0], c[1]) for c in claims]
    added = []
    for m in REGION.finditer(text):  # regions first: "escapes to the center" is then a flight with a region
        if not any(a < m.end() and b > m.start() for a, b in taken):
            mentions.append([m.start(), m.end(), "REGION", None, None, None])
            taken.append((m.start(), m.end()))
            added.append(("REGION", m.group()))
    for m in PRESSURE.finditer(text):
        s, e = m.start(), m.end()
        stop = e + (re.search(r"[.;:!?\n,]|\band\b|\bbut\b|\bwhile\b|$", text[e:]).start())
        target = any(x[2] in ("SQUARE", "PIECE", "REGION", "FILE", "RANK", "DIAGONAL") and e <= x[0] < stop for x in mentions)
        subject = any(x[2] in ("PIECE", "MOVE") and x[1] <= s and not re.search(r"[.;!?\n]", text[x[1]:s]) for x in mentions)
        if target and subject and not any(a < e and b > s for a, b in taken):
            claims.append([s, e, "ATTACKS", "NOW" if first_move is None or s < first_move else "AFTER"])
            taken.append((s, e))
            added.append(("ATTACKS", text[s:e]))
    for pattern, kind in PATTERNS:
        for m in pattern.finditer(text):
            s, e = m.start(), m.end()
            span = text[s:e]
            if kind == "FLIGHT":
                if not KING_BEFORE.search(text[max(0, s - 40):s]) or SQUARE_AFTER.match(text[e:]) or \
                        NOUN_BEFORE.search(text[max(0, s - 6):s]) or NOUN_AFTER.match(text[e:]):
                    continue
                king = next((x for x in mentions if x[2] == "PIECE" and re.search(r"king", text[x[0]:x[1]], re.I)
                             and 0 <= s - x[1] <= 35), None)
                if king is None:  # the king is the claim's subject: label it if the labeller left it out
                    words = [w for w in KING_WORD.finditer(text, max(0, s - 35), s)
                             if not re.search(r"[.;:,]", text[w.end():s])]
                    if not words or any(a < words[-1].end() and b > words[-1].start() for a, b in taken):
                        continue
                    w = words[-1]
                    if w.group().lower().startswith("the "):
                        continue
                    mentions.append([w.start(), w.end(), "PIECE", None, None, "AFTER"])
                    taken.append((w.start(), w.end()))
                    added.append(("PIECE", w.group()))
                lead = re.match(r"(?:then|just)\s+", span, re.I)  # "then escapes": "then" stays outside
                if lead:
                    s += lead.end()
            if kind == "EVAL_SIDE":
                if KING_BEFORE.search(text[max(0, s - 40):s]):
                    continue  # "the king ... and escapes" is flight, handled above if it fits
                s += len(re.match(r"(?:and|to)\s+(?:simply\s+)?", span, re.I).group())
                kind = "EVAL"
            if kind == "EVAL" and re.match(r"(?:the|a)\s", text[s:e], re.I):
                s += len(re.match(r"(?:the|a)\s+", text[s:e], re.I).group())
            if first_move is None or s < first_move:
                continue  # a refutation answers a move
            if any(a < e and b > s for a, b in taken):
                continue
            claims.append([s, e, kind, "AFTER"])
            taken.append((s, e))
            added.append((kind, text[s:e]))
    if not added:
        return marked, []
    claims.sort()
    mentions.sort()
    new = render(text, mentions, claims)
    assert parse(new)["text"] == text
    return new, added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--file", action="append", help="relabel these files instead of the dataset's labels")
    ap.add_argument("--key", default="marked")
    ap.add_argument("--samples", type=int, default=25)
    args = ap.parse_args()
    files = args.file or (sorted(glob.glob(str(ROOT / "data/reflections/labels/*.jsonl"))) +
                          [f for f in sorted(glob.glob(str(ROOT / "data/reflections/synthetic/*.jsonl"))) if "templated" not in f])
    counts, samples = Counter(), []
    for f in files:
        rows = [json.loads(line) for line in Path(f).read_text().splitlines() if line.strip()]
        changed = False
        for r in rows:
            new, added = relabel(r[args.key])
            if added:
                changed = True
                r[args.key] = new
                for kind, t in added:
                    counts[(kind, t.lower())] += 1
                samples.append((Path(f).name, r.get("id"), new))
        if changed and not args.dry_run:
            Path(f).write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(sum(counts.values()), "claims added")
    for (kind, t), n in counts.most_common(60):
        print(f"  {n:4d} {kind} {t}")
    random.Random(0).shuffle(samples)
    for name, rid, new in samples[:args.samples]:
        print(f"\n{name} {rid}\n{new}")


if __name__ == "__main__":
    main()
