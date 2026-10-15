"""The v3 markup: human-readable labels written inline in the text.

Mentions are wrapped in braces, with their type and attributes in square brackets:

    {Qxc3+}[move new rejected]   a move. line: new | next | alt   stance: chosen | rejected | considered
    {queen}[piece now]           a piece (with its colour if attached). frame: now | after
    {e4}[square]  {c-file}[file]  {7th rank}[rank]  {a1-h8}[diagonal]

Claims are wrapped in angle brackets, with their type and frame:

    <gives check>[check after]  <checkmate>[mate after]  <wins>[wins after]
    <covers>[attacks now]  <cannot go to>[flight after]

line:  new  = played from the current (puzzle) position
       next = played after the move mentioned just before it (continues that line)
       alt  = played in the same position as the move mentioned just before it
              (another reply to the same move: a sibling, not a continuation)
frame: now   = true/false in the current position, before any move
       after = in the position after the last move mentioned before it

parse() turns marked text into plain text plus character spans:

    {"text": ...,
     "mentions": [[start, end, type, line, stance, frame], ...],   # None where not applicable
     "claims": [[start, end, type, frame], ...]}

The frame may be missing (None) in older labels; check(..., frames=True) requires it.
"""

import re

MENTION_TYPES = ["MOVE", "PIECE", "SQUARE", "FILE", "RANK", "DIAGONAL"]
CLAIM_TYPES = ["CHECK", "MATE", "WINS", "ATTACKS", "FLIGHT"]
LINES = ["NEW", "NEXT", "ALT"]
STANCES = ["CHOSEN", "REJECTED", "CONSIDERED"]
FRAMES = ["NOW", "AFTER"]

TOKEN = re.compile(r"\{([^{}]+)\}\[([a-z ]+)\]|<([^<>]+)>\[([a-z ]+)\]")


class MarkupError(ValueError):
    pass


def _frame(attrs, what):
    if len(attrs) == 0:
        return None
    if len(attrs) == 1 and attrs[0].upper() in FRAMES:
        return attrs[0].upper()
    raise MarkupError(f"expected an optional frame now|after: {what!r}")


def parse(marked: str) -> dict:
    text, mentions, claims = [], [], []
    cursor = length = 0
    for m in TOKEN.finditer(marked):
        before = marked[cursor:m.start()]
        inner = m.group(1) if m.group(1) is not None else m.group(3)
        attrs = (m.group(2) if m.group(1) is not None else m.group(4)).split()
        start = length + len(before)
        end = start + len(inner)
        if inner != inner.strip():
            raise MarkupError(f"span has leading/trailing space: {inner!r}")
        if m.group(1) is not None:
            kind = attrs[0].upper()
            if kind not in MENTION_TYPES:
                raise MarkupError(f"unknown mention type {attrs[0]!r} in {m.group(0)!r}")
            if kind == "MOVE":
                if len(attrs) != 3 or attrs[1].upper() not in LINES or attrs[2].upper() not in STANCES:
                    raise MarkupError(f"a move needs [move new|next|alt chosen|rejected|considered]: {m.group(0)!r}")
                mentions.append([start, end, kind, attrs[1].upper(), attrs[2].upper(), None])
            elif kind == "PIECE":
                mentions.append([start, end, kind, None, None, _frame(attrs[1:], m.group(0))])
            else:
                if len(attrs) != 1:
                    raise MarkupError(f"only moves and pieces take attributes: {m.group(0)!r}")
                mentions.append([start, end, kind, None, None, None])
        else:
            if not attrs or attrs[0].upper() not in CLAIM_TYPES:
                raise MarkupError(f"unknown claim type in {m.group(0)!r}")
            claims.append([start, end, attrs[0].upper(), _frame(attrs[1:], m.group(0))])
        text += [before, inner]
        length = end
        cursor = m.end()
    text.append(marked[cursor:])
    plain = "".join(text)
    if re.search(r"\}\[|>\[", plain):
        raise MarkupError("unparsed markup left in the text (check the brackets)")
    return {"text": plain, "mentions": mentions, "claims": claims}


def render(text: str, mentions: list, claims: list) -> str:
    """The inverse of parse()."""
    items = [(m[0], m[1], "m", m[2:]) for m in mentions] + [(c[0], c[1], "c", c[2:]) for c in claims]
    out, cursor = [], 0
    for start, end, which, rest in sorted(items, key=lambda x: (x[0], x[1])):
        if start < cursor:
            raise MarkupError(f"overlapping spans at {start}")
        out.append(text[cursor:start])
        if which == "m":
            kind, line, stance, frame = (list(rest) + [None] * 4)[:4]
            attrs = [kind]
            if kind == "MOVE":
                attrs += [line, stance]
            elif kind == "PIECE" and frame:
                attrs.append(frame)
            out.append("{" + text[start:end] + "}[" + " ".join(a.lower() for a in attrs) + "]")
        else:
            kind, frame = (list(rest) + [None])[:2]
            attrs = [kind] + ([frame] if frame else [])
            out.append("<" + text[start:end] + ">[" + " ".join(a.lower() for a in attrs) + "]")
        cursor = end
    return "".join(out + [text[cursor:]])


def check(marked: str, original: str | None = None, frames: bool = False) -> list[str]:
    """Problems with one labelled example (empty list = fine)."""
    try:
        parsed = parse(marked)
    except MarkupError as e:
        return [str(e)]
    problems = []
    if original is not None and parsed["text"] != original:
        a, b = parsed["text"], original
        i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
        problems.append(f"text differs from the original at char {i}: got {a[i:i+40]!r}, expected {b[i:i+40]!r}")
    moves = [m for m in parsed["mentions"] if m[2] == "MOVE"]
    if moves and moves[0][3] != "NEW":
        problems.append(f"the first move is {moves[0][3]} (nothing to continue)")
    if frames:
        text = parsed["text"]
        missing = [text[m[0]:m[1]] for m in parsed["mentions"] if m[2] == "PIECE" and m[5] is None]
        missing += [text[c[0]:c[1]] for c in parsed["claims"] if c[3] is None]
        if missing:
            problems.append(f"no frame (now|after) on: {missing[:6]}")
    return problems
