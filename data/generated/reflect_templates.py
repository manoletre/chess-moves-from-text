"""Multi-PV engine lines -> a varied but true reflection. Every move is named, and every claim comes from the row or the
board. The same row always gives the same text (seeded by id). Word pools never claim more than the board shows."""
import random
import zlib
from collections import Counter

import chess

NAME = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop", chess.ROOK: "rook", chess.QUEEN: "queen",
        chess.KING: "king"}
VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}

OPENERS = ["Let me compare a few moves.", "A few moves stand out here.", "What are my options?", "My turn, so let me think.",
           "I'll look at the main candidates.", "Let me calculate some lines.", "Time to work through this.",
           "Which move holds up best?", "Let me test a few ideas.", "I want to check some concrete lines.",
           "Before moving, let me look deeper.", "Several moves deserve a look.", "Let me see what each move leads to."]
TRY_FIRST = ["I could try {m}.", "What about {m}?", "Consider {m}.", "One option is {m}.", "Start with {m}.",
             "First idea: {m}.", "Let me look at {m}.", "How about {m}?", "I'm tempted by {m}.", "A natural try is {m}."]
TRY_NEXT = TRY_FIRST[:4] + ["Alternatively, there is {m}.", "Another idea is {m}.", "Or I could look at {m}.",
                            "There is also {m}.", "Next up: {m}.", "What if I go for {m} instead?",
                            "On the other hand, I have {m}.", "Then again, {m} is possible."]
LEAD = ["Then", "After that,", "From there,", "In that line,", "Most likely,", "Next,", "Following that,", "If so,"]
JOIN = [", then ", "; then ", ", and then ", ", after which ", "; next, ", ". Then ", ". After that, "]
WHY = ["{m} holds up best, while {o} is worse.", "{m} beats {o} here.", "{o} falls short of {m}.",
       "Between them, {m} is stronger than {o}.", "I prefer {m} to {o}.", "{o} gives me less than {m}.",
       "{m} comes out ahead of {o}."]
END = ["Best is {m}.", "So I'll go with {m}.", "That makes {m} my choice.", "I choose {m}.", "The pick is {m}.",
       "I've settled on {m}.", "My decision is {m}.", "That settles it: {m}.", "So the move is {m}.",
       "Overall, {m} looks strongest.", "I'm going with {m}.", "Comparing all that, I pick {m}."]
BANDS = [(500, ["I'd be winning", "that wins for me", "I come out far ahead", "my position would be winning",
                "that should be decisive in my favor", "I'd have a winning game"]),
         (150, ["I'd be clearly better", "that gives me a real edge", "I'd have a solid advantage",
                "that leaves me clearly on top", "my position would be clearly better"]),
         (50, ["I'd be slightly better", "that keeps a small edge", "I'd have a little something",
               "that gives me a modest plus", "I'd stay a bit better"]),
         (-50, ["it's about equal", "the game stays balanced", "neither side has much", "that looks level",
                "the position is roughly even", "things stay about even"]),
         (-150, ["I'd be slightly worse", "that leaves me a little worse", "they'd have a small edge",
                 "I'd be a bit on the back foot", "they'd keep a modest plus"]),
         (-500, ["I'd be clearly worse", "that leaves me in trouble", "they'd have a clear advantage",
                 "my position would be clearly worse", "they'd be clearly on top"]),
         (-10 ** 9, ["I'd be losing", "that loses for me", "my position would be lost", "they'd have a winning game",
                     "that should be decisive against me"])]


TIERS = [(10.0, "hard", (4, 5), (7, 10)), (3.0, "medium", (3, 5), (5, 7)), (-1e9, "easy", (2, 4), (2, 5))]


def tier(row):
    """Difficulty from how far the best move stands out (win% over the second best): one good move = hard."""
    gap = row["multipv"][0]["win"] - row["multipv"][1]["win"]
    return next(t for t in TIERS if gap >= t[0])


def band(c):
    if c["mate"] is not None:
        return f"mate{c['mate']}"
    return next(lo for lo, _ in BANDS if c["cp"] >= lo)


def verdict(c, rng, numbers):
    if c["mate"] is not None:
        n = abs(c["mate"])
        pool = ([f"I mate in {n}", f"that's mate in {n} for me", f"I have a forced mate in {n}"] if c["mate"] > 0 else
                [f"I get mated in {n}", f"they mate me in {n}", f"that runs into mate in {n}"])
        return rng.choice(pool)
    text = next(rng.choice(p) for lo, p in BANDS if c["cp"] >= lo)
    num = f"{c['cp'] / 100:+.1f}"
    if numbers or rng.random() < 0.3:
        return rng.choice([f"{text} ({num})", f"{text}, about {num}", f"{text}, roughly {num}"])
    return text


def quiet_verbs(pt):
    """Verb frames for a move without a capture: (gerund, first person, third person)."""
    v = [("moving", "move"), ("bringing", "bring"), ("playing", "play"), ("sending", "send"), ("shifting", "shift")]
    if pt == chess.PAWN:
        v = [("pushing", "push"), ("advancing", "advance"), ("moving", "move"), ("playing", "play")]
    elif pt == chess.KNIGHT:
        v += [("jumping", "jump"), ("hopping", "hop")]
    elif pt in (chess.BISHOP, chess.ROOK, chess.QUEEN):
        v += [("sliding", "slide"), ("swinging", "swing")]
    elif pt == chess.KING:
        v = [("moving", "move"), ("stepping", "step"), ("walking", "walk")]
    return v


def gerund(board, m, rng):
    """'moving my bishop to g6', from the mover's point of view."""
    pt, to = board.piece_type_at(m.from_square), chess.square_name(m.to_square)
    p = NAME[pt]
    if board.is_kingside_castling(m):
        return rng.choice(["castling kingside", "castling short", "getting my king safe by castling kingside"])
    if board.is_queenside_castling(m):
        return rng.choice(["castling queenside", "castling long", "castling to the queenside"])
    if m.promotion:
        q = NAME[m.promotion]
        return rng.choice([f"promoting my pawn to a {q} on {to}", f"making a {q} on {to}", f"getting a new {q} on {to}"])
    if board.is_en_passant(m):
        return rng.choice([f"taking the pawn en passant, landing on {to}", f"capturing en passant on {to}"])
    if board.is_capture(m):
        v = NAME[board.piece_type_at(m.to_square)]
        verb, base = rng.choice([("taking", "take"), ("capturing", "capture"), ("grabbing", "grab"), ("removing", "remove"),
                                 ("eating", "eat"), ("snapping up", "snap up"), ("picking off", "pick off"),
                                 ("knocking out", "knock out"), ("gobbling up", "gobble up"), ("wiping out", "wipe out")])
        return rng.choice([f"{verb} the {v} on {to} with my {p}", f"using my {p} to {base} the {v} on {to}",
                           f"my {p} {verb} the {v} on {to}"])
    verb = rng.choice(quiet_verbs(pt))[0]
    return rng.choice([f"{verb} my {p} to {to}", f"{verb} the {p} to {to}", f"putting my {p} on {to}",
                       f"the {p} going to {to}", f"my {p} heading for {to}"])


def clause(board, m, mine, rng):
    """'their knight goes to f6' or 'I take the pawn on e5 with my knight'. Says 'with check' only when it is check."""
    pt, to = board.piece_type_at(m.from_square), chess.square_name(m.to_square)
    p, who, own = NAME[pt], "I" if mine else "they", "my" if mine else "their"
    if board.is_castling(m):
        side = "kingside" if board.is_kingside_castling(m) else "queenside"
        text = rng.choice([f"{who} castle {side}", f"{who} castle to the {side}"])
    elif m.promotion:
        text = f"{who} promote to a {NAME[m.promotion]} on {to}"
    elif board.is_en_passant(m):
        text = f"{who} take the pawn en passant on {to}"
    elif board.is_capture(m):
        v = NAME[board.piece_type_at(m.to_square)]
        base, third = rng.choice([("take", "takes"), ("capture", "captures"), ("grab", "grabs"), ("remove", "removes"),
                                  ("eat", "eats"), ("snap up", "snaps up"), ("pick off", "picks off"),
                                  ("knock out", "knocks out"), ("gobble up", "gobbles up"), ("wipe out", "wipes out")])
        text = rng.choice([f"{who} {base} the {v} on {to} with {own} {p}", f"{own} {p} {third} the {v} on {to}",
                           f"{who} use {own} {p} to {base} the {v} on {to}"])
    else:
        verb = rng.choice(quiet_verbs(pt))[1]
        text = rng.choice([f"{who} {verb} {own} {p} to {to}", f"{own} {p} goes to {to}", f"{own} {p} heads to {to}",
                           f"{own} {p} lands on {to}", f"{own} {p} moves to {to}", f"{own} {p} comes to {to}",
                           f"{who} put {own} {p} on {to}", f"{who} answer with {own} {p} to {to}"
                           if not mine else f"{who} follow up with {own} {p} to {to}"])
    after = board.copy()
    after.push(m)
    if after.is_check() and not after.is_checkmate() and rng.random() < 0.5:
        text += ", giving check" if " with " in text else " with check"
    return text


def named(text, san, rng):
    return f"{text} ({san})" if rng.random() < 0.85 else f"{san}, {text}"


def material(board, color):
    return sum(v * (len(board.pieces(pt, color)) - len(board.pieces(pt, not color))) for pt, v in VALUE.items())


def facts(board, m, rng):
    """True statements about a candidate move, checked on the board: a list of (kind, text)."""
    out, me, pt = [], board.turn, board.piece_type_at(m.from_square)
    after = board.copy()
    after.push(m)
    moved = m.promotion or pt
    if after.is_checkmate():
        return [("mate", rng.choice(["It is checkmate.", "That is mate on the board.", "The game ends right there, checkmate."]))]
    if after.is_check():
        out.append(("check", rng.choice(["It gives check.", "That comes with check.", "That's check.",
                                         "Their king ends up in check."])))
    targets = []
    if moved != chess.KING:
        for sq in after.attacks(m.to_square):
            q = after.piece_at(sq)
            if not q or q.color == me or q.piece_type == chess.KING:
                continue
            undefended = not after.attackers(not me, sq)
            if undefended or VALUE[q.piece_type] > VALUE[moved]:
                targets.append((VALUE[q.piece_type], f"the {'undefended ' if undefended else ''}"
                                f"{NAME[q.piece_type]} on {chess.square_name(sq)}"))
    targets.sort(reverse=True)
    if len(targets) >= 2:
        a, b = targets[0][1], targets[1][1]
        out.append(("fork", rng.choice([f"It attacks both {a} and {b}.", f"That hits {a} and {b} at once.",
                                        f"Now {a} and {b} are both attacked."])))
    elif targets:
        t = targets[0][1]
        out.append(("attack", rng.choice([f"It attacks {t}.", f"It hits {t}.", f"That puts {t} under attack.",
                                          f"It goes after {t}.", f"Now {t} is attacked."])))
    hits = [a for a in after.attackers(not me, m.to_square) if after.is_legal(chess.Move(a, m.to_square))]
    cheaper = [a for a in hits if VALUE[after.piece_type_at(a)] < VALUE[moved]]
    if cheaper and moved != chess.KING:
        a = f"their {NAME[after.piece_type_at(cheaper[0])]} on {chess.square_name(cheaper[0])}"
        out.append(("risk", rng.choice([f"But {a} can take it.", f"The catch is that {a} can capture it.",
                                        f"However, {a} can take it."])))
    elif not hits and board.is_attacked_by(not me, m.from_square) and moved != chess.KING and pt != chess.PAWN:
        out.append(("safe", rng.choice([f"It also moves my {NAME[pt]} out of an attack.",
                                        f"That takes my {NAME[pt]} out of an attack.",
                                        f"My {NAME[pt]} also escapes an attack that way."])))
    return out


def line_text(board, pv, rng, plies):
    """The plies after the candidate, each named. Returns the sentence, the SANs and the material change."""
    b, me, bits, sans = board.copy(), board.turn, [], []
    b.push(chess.Move.from_uci(pv[0]))
    for u in pv[1:1 + min(rng.randint(*plies), len(pv) - 1)]:
        m = chess.Move.from_uci(u)
        sans.append(b.san(m))
        bits.append(f"{clause(b, m, b.turn == me, rng)} ({sans[-1]})")
        b.push(m)
    if not bits:
        return "", sans, 0
    lead = rng.choice(LEAD)
    text, last = lead + " " + bits[0], lead.lower()
    for bit in bits[1:]:
        j = rng.choice([x for x in JOIN if not any(w in x.lower() and w in last for w in ("next", "then", "after"))])
        text += j + bit
        last = j.lower()
    return text + ".", sans, material(b, me) - material(board, me)


def material_text(delta, rng):
    n = abs(delta)
    if delta > 0:
        return rng.choice([f"That leaves me up {n} in material.", f"I'd have gained {n} in material by then.",
                           f"By then I'm {n} up on material compared with now."])
    return rng.choice([f"That leaves me down {n} in material.", f"I'd have lost {n} in material by then.",
                       f"By then I'm {n} down on material compared with now."])


def reflect(row):
    """Returns {text, best, order, cands}. `cands` holds each candidate's pieces for the metrics."""
    rng = random.Random(zlib.crc32(row["id"].encode()))
    board = chess.Board(row["fen"])
    _, level, cand_range, plies = tier(row)
    n = min(rng.randint(*cand_range), len(row["multipv"]))
    cands = rng.sample(row["multipv"][:n], n)      # the best one can sit anywhere
    bands = Counter(band(c) for c in cands)
    parts = [rng.choice(OPENERS)] if rng.random() < 0.8 else []
    sans, info = {}, []
    for i, c in enumerate(cands):
        m = chess.Move.from_uci(c["pv"][0])
        san = board.san(m)
        sans[c["uci"]] = san
        intro = rng.choice(TRY_NEXT if i else TRY_FIRST).format(m=named(gerund(board, m, rng), san, rng))
        fs = facts(board, m, rng)
        line, line_sans, delta = line_text(board, c["pv"], rng, plies)
        mat = material_text(delta, rng) if abs(delta) >= 1 else ""
        v = verdict(c, rng, numbers=bands[band(c)] > 1)
        v = v[0].upper() + v[1:] + "."
        fact_text = " ".join(t for _, t in fs)
        parts.append(" ".join(p for p in [intro, fact_text, line, mat, v] if p))   # facts right after the move they're about
        info.append({"uci": c["uci"], "san": san, "sans": line_sans, "core": intro + " " + line, "facts": fs,
                     "material": mat, "delta": delta, "verdict": v, "cand": c})
    best = row["multipv"][0]
    other = max((c for c in cands if c["uci"] != best["uci"]), key=lambda c: c["win"])
    if best["win"] - other["win"] > 3 and rng.random() < 0.6:
        parts.append(rng.choice(WHY).format(m=sans[best["uci"]], o=sans[other["uci"]]))
    final = rng.choice(END).format(m=named(gerund(board, chess.Move.from_uci(best["uci"]), rng), sans[best["uci"]], rng))
    parts.append(final)
    return {"text": " ".join(p for p in parts if p), "best": best["uci"], "order": [c["uci"] for c in cands],
            "cands": info, "final": final, "tier": level}
