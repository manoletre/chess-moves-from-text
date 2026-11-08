"""Gold labels for programmatically generated reflections (data/generated/reflect_templates.py, a snapshot).

The generator isn't changed: `labelled(row)` runs it, and because its output says which move, fact and verdict each
candidate has, the labels can be read back from the text. The text itself is treated as correct.

    python scripts/generated_labels.py --rows ~/Dev/chessLLM/data/d1/games_100k.jsonl --mode natural --games test --n 1000 --out x.jsonl
    python scripts/generated_labels.py --check     # guards: same text as the generator, held-out phrases really held out

How the text maps to the v3 labels (LABELING.md):
- A move named in prose and SAN ("my knight heads to e5 (Ne5)") is one MOVE, the SAN only (section 4); the prose around it
  gets no labels at all: no pieces, squares or claims. It restates the move, and verify.py would read "my knight on e5"
  as a placement in the position before the move.
- The best candidate's line is CHOSEN, every other candidate's line REJECTED (they lose to the choice at the end).
- "their knight on f6 can take it" is a reply stated as possible, so a MOVE; "their king on e8 can take it" is a FLIGHT
  claim (section 6). A reply after it in the line is then ALT: both answer the candidate.
- Facts become claims with their pieces and squares; verdicts ("I'd be clearly better") EVAL, "mate in N" verdicts MATE.
- Material counts ("That leaves me up 3 in material") stay unlabelled, like "material is even" (section 4).

Held-out wording: `mode` restricts the generator's own random choices, not its code.
    natural   the generator exactly as it is (byte-identical text)
    train     never picks a phrase in HELD_OUT
    heldout   picks a phrase in HELD_OUT wherever the pool has one
"""

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data" / "generated"))
import reflect_templates as T  # noqa: E402

from chess_tagger.markup import check, render  # noqa: E402

PIECE = r"pawn|knight|bishop|rook|queen|king"
SQ = r"[a-h][1-8]"
CAPTURE_VERBS = ["taking", "capturing", "grabbing", "removing", "eating", "snapping up", "picking off", "knocking out",
                 "gobbling up", "wiping out", "take", "takes", "capture", "captures", "grab", "grabs", "remove", "removes",
                 "eat", "eats", "snap up", "snaps up", "pick off", "picks off", "knock out", "knocks out", "gobble up",
                 "gobbles up", "wipe out", "wipes out"]
QUIET_VERBS = ["moving", "bringing", "playing", "sending", "shifting", "pushing", "advancing", "jumping", "hopping", "sliding",
               "swinging", "stepping", "walking"]


def phrase(choice) -> str:
    """What a random choice says once its pieces, squares and numbers are masked: the key held-out wording is listed by."""
    if isinstance(choice, tuple):  # ("moving", "move"), ("take", "takes"): a verb
        return choice[0]
    if not isinstance(choice, str):
        return ""
    s = choice.replace("undefended ", "")
    s = re.sub(rf"\b(?:{PIECE})\b", "P", s)
    s = re.sub(rf"\b{SQ}\b", "S", s)
    s = re.sub(r"\b\d+\b", "N", s)
    s = re.sub(r"\b(?:" + "|".join(sorted(CAPTURE_VERBS, key=len, reverse=True)) + r")\b", "V", s)
    s = re.sub(r"^(?:" + "|".join(QUIET_VERBS) + r") (my|the) P to S$", r"V \1 P to S", s)
    s = re.sub(r"\b(?:my|their)\b(?= P)", "O", s)
    return re.sub(r"^(?:I|they)(?= )", "W", s)


# About a quarter of each pool. Every pool keeps phrases on the train side.
HELD_OUT = {
    # OPENERS
    "What are my options?", "Let me test a few ideas.", "Several moves deserve a look.",
    # TRY_FIRST, TRY_NEXT (their shared first four stay in train)
    "Start with {m}.", "I'm tempted by {m}.", "A natural try is {m}.",
    "What if I go for {m} instead?", "On the other hand, I have {m}.", "Then again, {m} is possible.",
    # LEAD, JOIN
    "Most likely,", "Following that,", ", after which ", "; next, ",
    # WHY, END
    "{o} gives me less than {m}.", "{m} comes out ahead of {o}.",
    "That settles it: {m}.", "My decision is {m}.", "Comparing all that, I pick {m}.",
    # verdict bands, one or two each, and mate verdicts
    "that should be decisive in my favor", "W come out far ahead", "that leaves me clearly on top",
    "I'd have a little something", "neither side has much", "I'd be a bit on the back foot", "that leaves me in trouble",
    "that should be decisive against me", "that's mate in N for me", "that runs into mate in N",
    # verbs
    "sending", "swinging", "hopping", "walking", "advancing", "gobbling up", "wiping out", "knocking out",
    "gobble up", "wipe out", "knock out",
    # how a move is written in prose
    "the P going to S", "O P heading for S", "using O P to V the P on S", "O P lands on S", "O P comes to S",
    "W use O P to V the P on S", "castling long", "castling to the queenside", "getting a new P on S",
    "V en passant on S", "W castle to the kingside", "W castle to the queenside",
    # facts
    "That's check.", "The game ends right there, checkmate.", "That hits the P on S and the P on S at once.",
    "It goes after the P on S.", "Now the P on S is attacked.", "The catch is that O P on S can V it.",
    "My P also escapes an attack that way.",
}


class SplitRandom(random.Random):
    """random.Random whose choice() respects the held-out phrases. mode=natural behaves exactly like random.Random."""
    mode, used = "natural", None

    def choice(self, seq):
        if self.used is not None:
            self.used.update(phrase(x) for x in seq)
        if self.mode == "train":
            seq = [x for x in seq if phrase(x) not in HELD_OUT] or seq
        elif self.mode == "heldout":
            seq = [x for x in seq if phrase(x) in HELD_OUT] or seq
        return super().choice(seq)


class _Random:  # stands in for the `random` module inside the generator
    Random = SplitRandom


def generate(row: dict, mode: str = "natural") -> dict:
    SplitRandom.mode = mode
    T.random = _Random
    return T.reflect(row)


def _san_span(text: str, san: str, cursor: int, stop: int | None = None) -> tuple[int, int]:
    """The next '(san)' or 'san,' at or after the cursor, or the bare SAN: where a move is named after its prose."""
    bare = re.compile(r"(?<![\w-])" + re.escape(san) + r"(?![\w+#=-])")
    for m in bare.finditer(text, cursor, stop if stop is not None else len(text)):
        return m.start(), m.end()
    raise ValueError(f"{san!r} not found after {cursor}: {text[cursor:cursor + 80]!r}")


def _named_span(text: str, san: str, cursor: int) -> tuple[int, int]:
    """SAN in '... (san)' or 'san, ...', skipping the same characters inside the prose ('pawn to e4 (e4)')."""
    best = None
    for m in re.finditer(r"\(" + re.escape(san) + r"\)|(?<![\w-])" + re.escape(san) + r"(?=, )", text[cursor:]):
        start = cursor + m.start() + (m.group(0)[0] == "(")
        best = (start, start + len(san))
        break
    if best is None:
        raise ValueError(f"named {san!r} not found: {text[cursor:cursor + 80]!r}")
    return best


CHECK_FACT = re.compile(r"gives check|comes with check|(?<=That's )check|ends up in check")
MATE_FACT = re.compile(r"is checkmate|is mate|game ends right there, checkmate")
ATTACK_VERB = re.compile(r"(?<=^It )attacks both|(?<=^It )attacks|(?<=^It )hits|(?<=^That )hits|(?<=^It )goes after|under attack"
                         r"|is attacked|are both attacked")
SAFE = re.compile(r"out of an attack|escapes an attack")
RISK = re.compile(rf"their ({PIECE}) on ({SQ}) (can (?:take|capture) it)")
VERDICT_NUMBER = re.compile(r"(?: \([+-]\d+\.\d\)|, (?:about|roughly) [+-]\d+\.\d)?\.$")


def _fact_labels(kind: str, text: str, offset: int, stance: str, mentions: list, claims: list) -> bool:
    """Labels of one fact sentence, placed at `offset`. Returns True if it named a reply (the line's next move is then ALT)."""
    def claim(m, kind_):
        claims.append([offset + m.start(), offset + m.end(), kind_, "AFTER"])

    def piece(start, end):
        mentions.append([offset + start, offset + end, "PIECE", None, None, "AFTER"])

    def square(start, end):
        mentions.append([offset + start, offset + end, "SQUARE", None, None, None])

    if kind == "check":
        claim(CHECK_FACT.search(text), "CHECK")
        king = re.search(r"\bking\b", text)
        if king:
            piece(*king.span())
    elif kind == "mate":
        claim(MATE_FACT.search(text), "MATE")
    elif kind in ("attack", "fork"):
        claim(ATTACK_VERB.search(text), "ATTACKS")
        for m in re.finditer(rf"the (undefended )?({PIECE}) on ({SQ})", text):
            if m.group(1):
                claims.append([offset + m.start(1), offset + m.end(1) - 1, "ATTACKS", "AFTER"])
            piece(*m.span(2))
            square(*m.span(3))
    elif kind == "risk":
        m = RISK.search(text)
        if m.group(1) == "king":
            piece(*m.span(1))
            square(*m.span(2))
            claim(re.compile(r"can (?:take|capture) it").search(text), "FLIGHT")
            return False
        mentions.append([offset + m.start(1), offset + m.end(3), "MOVE", "NEXT", stance, None])
        return True
    elif kind == "safe":
        piece(*re.search(rf"\b(?:{PIECE})\b", text).span())
        claim(SAFE.search(text), "ATTACKS")
    else:
        raise ValueError(kind)
    return False


def labelled(row: dict, mode: str = "natural") -> dict:
    """One dataset line (the shape of data/*.jsonl) for a Stockfish row. `components` says what each span came from."""
    out = generate(row, mode)
    text, cursor = out["text"], 0
    mentions, claims = [], []

    def move(span, line, stance):
        mentions.append([span[0], span[1], "MOVE", line, stance, None])

    for c in out["cands"]:
        stance = "CHOSEN" if c["uci"] == out["best"] else "REJECTED"
        span = _named_span(text, c["san"], cursor)
        move(span, "NEW", stance)
        cursor = span[1]
        reply_named = False
        for kind, fact in c["facts"]:
            at = text.index(fact, cursor)
            reply_named |= _fact_labels(kind, fact, at, stance, mentions, claims)
            cursor = at + len(fact)
        for i, san in enumerate(c["sans"]):
            at = text.index(f"({san})", cursor) + 1
            for words in ("with check", "giving check"):  # said in the clause, just before the move's SAN
                if text[:at - 2].endswith(" " + words):
                    claims.append([at - 2 - len(words), at - 2, "CHECK", "AFTER"])
            move((at, at + len(san)), "ALT" if i == 0 and reply_named else "NEXT", stance)
            cursor = at + len(san)
        at = text.index(c["verdict"], cursor)
        words = VERDICT_NUMBER.sub("", c["verdict"])
        claims.append([at, at + len(words), "MATE" if c["cand"]["mate"] is not None else "EVAL", "AFTER"])
        cursor = at + len(c["verdict"])

    final_at = text.index(out["final"], cursor)
    if final_at > cursor:  # the comparison sentence ("Nf3 beats Bg6 here.")
        sans = {c["uci"]: c["san"] for c in out["cands"]}
        found = []
        for uci, san in sans.items():
            bare = re.search(r"(?<![\w-])" + re.escape(san) + r"(?![\w+#=-])", text[cursor:final_at])
            if bare:
                found.append((cursor + bare.start(), cursor + bare.end(), uci))
        for start, end, uci in sorted(found):
            move((start, end), "NEW", "CHOSEN" if uci == out["best"] else "REJECTED")
    span = _named_span(text, next(c["san"] for c in out["cands"] if c["uci"] == out["best"]), final_at)
    move(span, "NEW", "CHOSEN")

    mentions.sort()
    claims.sort()
    return {"text": text, "mentions": mentions, "claims": claims, "labelled": "full", "marked": render(text, mentions, claims),
            "group": "generated", "source": f"generated:{mode}:{out['tier']}", "id": row["id"], "fen": row["fen"],
            "played": out["best"], "tier": out["tier"]}


def game_split(game_id: str) -> str:
    """Games, not positions, are split, so no game is in two of train / val / test."""
    n = int(hashlib.md5(game_id.encode()).hexdigest(), 16) % 10
    return "val" if n == 0 else "test" if n == 1 else "train"


def read_rows(path, games: str | None, n: int, seed: int = 0) -> list:
    """Up to n rows spread over the file, from the games of one split, one position per game when possible."""
    with open(path) as f:
        lines = f.readlines()
    rng = random.Random(seed)
    rng.shuffle(lines)
    rows, seen = [], set()
    for line in lines:
        r = json.loads(line)
        if len(r["multipv"]) < 3 or not all(c.get("pv") for c in r["multipv"]):  # same filter as reflect_metrics.py
            continue
        if games and game_split(r["game_id"]) != games:
            continue
        if r["game_id"] in seen:  # one position per game keeps the sample varied
            continue
        seen.add(r["game_id"])
        rows.append(r)
        if len(rows) == n:
            break
    return rows


def self_check(path) -> None:
    rows = read_rows(path, None, 3000)
    SplitRandom.used = set()
    train_used, bad = set(), 0
    for r in rows:
        original = random.Random
        T.random = random
        plain = T.reflect(r)["text"]
        T.random = _Random
        e = labelled(r, "natural")
        assert e["text"] == plain, "natural mode changed the generator's text"
        problems = check(e["marked"], e["text"], frames=True)
        assert not problems, problems
    pools = set(SplitRandom.used)
    missing = HELD_OUT - pools
    print(f"natural: {len(rows)} texts identical to the generator, labels valid")
    print(f"phrases seen in pools: {len(pools)}, held out: {len(HELD_OUT)}, listed but never in a pool: {sorted(missing)}")
    # what train and heldout actually pick
    picked = {"train": set(), "heldout": set()}
    for mode in picked:
        class Spy(SplitRandom):
            def choice(self, seq):
                x = super().choice(seq)
                picked[mode].add(phrase(x))
                return x
        spy = type("_", (), {"Random": Spy})
        SplitRandom.used = None
        Spy.mode = mode
        for r in rows[:1500]:
            T.random = spy
            T.reflect(r)
    print("train picks a held-out phrase:", sorted(picked["train"] & HELD_OUT))
    print("heldout never picks:", sorted(HELD_OUT - picked["heldout"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=str(Path.home() / "Dev/chessLLM/data/d1/games_100k.jsonl"))
    ap.add_argument("--mode", choices=["natural", "train", "heldout"], default="natural")
    ap.add_argument("--games", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        self_check(a.rows)
    else:
        lines = [json.dumps(labelled(r, a.mode)) for r in read_rows(a.rows, a.games, a.n)]
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text("\n".join(lines) + "\n")
        print(f"{len(lines)} texts -> {a.out}")
