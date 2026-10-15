"""Check a tagged reflection against the position with python-chess.

    from chess_tagger import ChessTagger
    from chess_tagger.verify import verify

    tagger = ChessTagger("chess-move-tagger")
    report = verify(fen, text, tagger.predict([text])[0], played="d1d2")
    report["facts"]      # [{"kind": "mate", "text": "delivering checkmate", "verdict": False, ...}, ...]
    report["summary"]    # counts of true / false / unverifiable per kind, chosen move, matches_played

It also works on gold labels: verify(fen, parsed["text"], parsed) with parsed = markup.parse(marked).

Rules (the tagger finds the spans; everything here is deterministic):
- Moves are replayed as lines: NEW from the puzzle position, NEXT after the previous move,
  ALT from the same position as the previous move. SAN, UCI, long algebraic/arrows and prose
  ("the queen slides to h8", "capture the pawn on d7") are resolved against the legal moves.
- A claim or piece is checked in its frame: NOW = puzzle position, AFTER = after the last move
  mentioned before it. If that move couldn't be resolved, the fact is unverifiable.
- A claim's subject is the nearest MOVE/PIECE before it in its clause; its objects are the
  SQUARE/PIECE mentions after it in the clause. Negation is read from the claim span.

A fact's verdict is True, False or None (unverifiable). Unverifiable is NOT a safe
neutral: a reward should treat a high share of unverifiable claims as a smell.
"""

import re

import chess

PIECES = {"pawn": chess.PAWN, "knight": chess.KNIGHT, "bishop": chess.BISHOP,
          "rook": chess.ROOK, "queen": chess.QUEEN, "king": chess.KING}
PIECE_WORD = re.compile(r"\b(pawn|knight|bishop|rook|queen|king)s?\b", re.I)
SQUARE = re.compile(r"\b([a-h][1-8])\b")
NEGATION = re.compile(r"\b(not|no|cannot|can't|can not|never|without|nowhere|nothing|none|un\w+|longer)\b", re.I)
RESTRICTIVE = re.compile(r"box|trap|hem|seal|block|cut off|stuck|cramp|occup|no escape|nowhere|cannot|can't|no legal", re.I)
CAPTURE_VERB = re.compile(r"\b(captur\w*|takes?|taking|recaptur\w*|grab\w*|win\w*|snap\w*)\b", re.I)
PROMOTION = re.compile(r"promot\w*\s+(?:to\s+)?(?:an?\s+)?(queen|rook|bishop|knight)|=([QRBN])", re.I)
CASTLE = re.compile(r"castl\w*\s*(short|long|kingside|queenside|king-side|queen-side)?|\b([0O]-[0O](?:-[0O])?)\b", re.I)
UCI = re.compile(r"^([a-h][1-8])([a-h][1-8])([qrbn])?$")
TWO_SQUARES = re.compile(r"([a-h][1-8])\s*(?:-|x|→|->|to)\s*([a-h][1-8])")
CLAUSE_END = re.compile(r"[.;:!?\n]|,\s*(?:but|while|and then|so|because|as|whereas)\b|\bbut\b|\bbecause\b")


def square_of(text: str):
    """'h7' -> chess square; anything else (a mis-tagged span) -> None."""
    t = text.strip().lower()
    return chess.parse_square(t) if re.fullmatch(r"[a-h][1-8]", t) else None


def color_of(text: str):
    t = text.lower()
    if "white" in t:
        return chess.WHITE
    if "black" in t:
        return chess.BLACK
    return None


# ------------------------------------------------------------------ moves

def resolve(board: chess.Board, text: str):
    """Move text (any notation) -> (move | None, status) with status 'ok', 'illegal' or 'ambiguous'."""
    t = text.strip().replace("…", "").lstrip(".").strip("*`!? ")
    legal = list(board.legal_moves)
    if not legal:
        return None, "illegal"
    # castling
    m = CASTLE.search(t)
    if m:
        long = (m.group(2) and m.group(2).count("-") == 2) or (m.group(1) and m.group(1).lower().startswith("q"))
        target = "O-O-O" if long else "O-O"
        try:
            return board.parse_san(target), "ok"
        except ValueError:
            return None, "illegal"
    # UCI
    m = UCI.match(t)
    if m:
        move = chess.Move.from_uci(t)
        return (move, "ok") if move in legal else (None, "illegal")
    # SAN (with 0-0 and trailing + # annotations)
    if re.fullmatch(r"[KQRBNP]?[a-h]?[1-8]?x?[a-h][1-8](=?[QRBN])?[+#]?|O-O(-O)?[+#]?", t):
        try:
            return board.parse_san(t), "ok"
        except chess.AmbiguousMoveError:
            return None, "ambiguous"
        except ValueError:
            return None, "illegal"
    # long algebraic / arrows / prose: filter the legal moves by every constraint we can read
    candidates = legal
    two = TWO_SQUARES.search(t)
    squares = SQUARE.findall(t)
    if two:
        frm, to = chess.parse_square(two.group(1)), chess.parse_square(two.group(2))
        candidates = [mv for mv in candidates if mv.from_square == frm and mv.to_square == to]
    elif squares:
        to = chess.parse_square(squares[-1])
        candidates = [mv for mv in candidates if mv.to_square == to]
        if len(squares) >= 2:
            frm = chess.parse_square(squares[0])
            candidates = [mv for mv in candidates if mv.from_square == frm]
    letter = re.match(r"^([KQRBN])[a-h]", t)
    words = PIECE_WORD.findall(t)
    capture = bool(CAPTURE_VERB.search(t)) or bool(re.search(r"[a-h1-8]x[a-h]", t))
    promo = PROMOTION.search(t)
    if letter:
        candidates = [mv for mv in candidates if board.piece_type_at(mv.from_square) == "PNBRQK".index(letter.group(1)) + 1]
    elif words:
        first = PIECE_WORD.search(t)
        before = t[:first.start()]
        if CAPTURE_VERB.search(before):  # "capture the pawn on d7": the noun is the captured piece
            kind = PIECES[first.group(1).lower()]
            candidates = [mv for mv in candidates if board.piece_type_at(mv.to_square) == kind]
        elif not (promo and promo.start() < first.start()):
            kind = PIECES[first.group(1).lower()]
            candidates = [mv for mv in candidates if board.piece_type_at(mv.from_square) == kind]
            later = [w for w in PIECE_WORD.finditer(t) if w.start() > first.end()]
            if capture and later and not (promo and promo.start() < later[0].start()):
                victim = PIECES[later[0].group(1).lower()]
                candidates = [mv for mv in candidates if board.piece_type_at(mv.to_square) == victim]
    if capture:
        candidates = [mv for mv in candidates if board.is_capture(mv)]
    if promo:
        kind = PIECES[promo.group(1).lower()] if promo.group(1) else "PNBRQK".index(promo.group(2).upper()) + 1
        candidates = [mv for mv in candidates if mv.promotion == kind]
    if not (two or squares or letter or words or promo or capture):
        return None, "ambiguous"  # nothing to go on
    if len(candidates) == 1:
        return candidates[0], "ok"
    return None, "illegal" if not candidates else "ambiguous"


# ----------------------------------------------------------------- verify

def _clause_bounds(text: str, pos: int):
    start = 0
    for m in CLAUSE_END.finditer(text):
        if m.end() <= pos:
            start = m.end()
        elif m.start() >= pos:
            return start, m.start()
    return start, len(text)


def _piece_square(text: str, mention, mentions):
    """'queen on h7' -> h7 (the SQUARE right after a PIECE, joined by on/at/from)."""
    for m in mentions:
        if m[2] == "SQUARE" and m[0] >= mention[1] and re.fullmatch(r"\s*(on|at|from|in)\s*", text[mention[1]:m[0]]):
            return square_of(text[m[0]:m[1]])
    return None


def _find_piece(board, kind, color, square=None):
    if square is not None:
        return square
    squares = [sq for sq in chess.SQUARES if board.piece_type_at(sq) == kind
               and (color is None or board.color_at(sq) == color)]
    return squares[0] if len(squares) == 1 else None


def _king_moves(board: chess.Board, color) -> set:
    b = board.copy(stack=False)
    if b.turn != color:
        b.turn = color
        b.ep_square = None
    king = b.king(color)
    return {mv.to_square for mv in b.legal_moves if mv.from_square == king}


def verify(fen: str, text: str, tagged: dict, played: str | None = None) -> dict:
    root = chess.Board(fen)
    mentions = sorted(tagged["mentions"], key=lambda m: m[0])
    claims = sorted(tagged.get("claims") or [], key=lambda c: c[0])
    facts, moves = [], []  # moves: (start, end, before, move, after, stance)

    def fact(kind, span, verdict, detail=""):
        facts.append({"kind": kind, "start": span[0], "end": span[1], "text": text[span[0]:span[1]],
                      "verdict": verdict, "detail": detail})

    # 1. replay the lines
    prev = None  # (before, after) of the previous move
    for m in mentions:
        if m[2] != "MOVE":
            continue
        line = m[3] if prev else "NEW"
        before = root if line == "NEW" else (prev[1] if line == "NEXT" else prev[0])
        move, status, after = None, "unknown position", None
        if before is not None:
            move, status = resolve(before, text[m[0]:m[1]])
            if move is not None:
                after = before.copy(stack=False)
                after.push(move)
        moves.append((m[0], m[1], before, move, after, m[4]))
        prev = (before, after)
        if status == "ok":
            fact("legal_move", m, True, move.uci())
            notation = text[m[0]:m[1]]
            if notation.rstrip("!?").endswith("#"):
                fact("mate", m, after.is_checkmate(), "# in notation")
            elif notation.rstrip("!?").endswith("+"):
                fact("check", m, after.is_check(), "+ in notation")
            elif re.search(r"mate", notation, re.I):  # "rook on e2 delivers checkmate on e6"
                fact("mate", m, after.is_checkmate(), "mate inside the move")
            elif re.search(r"\bcheck", notation, re.I):
                fact("check", m, after.is_check(), "check inside the move")
            if re.search(r"[a-h1-8]x[a-h]", notation):
                fact("capture", m, before.is_capture(move), "x in notation")
        else:
            fact("legal_move", m, False if status == "illegal" else None, status)

    def frame_board(span, frame):
        if frame == "NOW":
            return root, None
        earlier = [mv for mv in moves if mv[1] <= span[0]]
        if not earlier:
            return root, None
        return earlier[-1][4], earlier[-1]

    # 2. piece placements ("queen on h7")
    for m in mentions:
        if m[2] != "PIECE":
            continue
        square = _piece_square(text, m, mentions)
        if square is None:
            continue
        board, _ = frame_board(m, m[5])
        word = PIECE_WORD.search(text[m[0]:m[1]])
        if board is None or word is None:
            fact("placement", m, None, "unknown position")
            continue
        kind, color = PIECES[word.group(1).lower()], color_of(text[m[0]:m[1]])
        ok = board.piece_type_at(square) == kind and (color is None or board.color_at(square) == color)
        fact("placement", (m[0], m[1] + len(text[m[1]:]) - len(text[m[1]:].lstrip()) + 5), ok,
             f"{chess.square_name(square)}: {board.piece_at(square) or 'empty'}")

    # 3. claims
    for c in claims:
        start, end, kind, frame = c[0], c[1], c[2], (c[3] if len(c) > 3 else None)
        board, last_move = frame_board(c, frame)
        if board is None:
            fact(kind.lower(), c, None, "unknown position")
            continue
        claim_text = text[start:end]
        negated = bool(NEGATION.search(claim_text))
        lo, hi = _clause_bounds(text, start)
        subject = next((m for m in reversed(mentions) if lo <= m[0] < start and m[2] in ("MOVE", "PIECE")), None)
        objects = [m for m in mentions if end <= m[0] < hi and m[2] in ("SQUARE", "PIECE")]

        if kind in ("CHECK", "MATE"):
            value = board.is_check() if kind == "CHECK" else board.is_checkmate()
            fact(kind.lower(), c, value != negated)
        elif kind == "WINS":
            mv = next((x for x in moves if subject and x[0] == subject[0]), last_move)
            if mv is None or mv[3] is None:
                fact("wins", c, None, "no resolved move")
                continue
            before, move = mv[2], mv[3]
            captured = before.piece_type_at(move.to_square) if before.is_capture(move) else None
            if before.is_en_passant(move):
                captured = chess.PAWN
            victims = [PIECES[w.group(1).lower()] for o in objects if o[2] == "PIECE"
                       for w in [PIECE_WORD.search(text[o[0]:o[1]])] if w]
            ok = captured is not None and (not victims or captured in victims)
            fact("wins", c, ok != negated, f"captures {chess.piece_name(captured) if captured else 'nothing'}")
        elif kind == "ATTACKS":
            if subject is None:
                fact("attacks", c, None, "no subject")
                continue
            if subject[2] == "MOVE":
                mv = next((x for x in moves if x[0] == subject[0]), None)
                src = mv[3].to_square if mv and mv[3] else None
                board = mv[4] if mv and mv[4] else board
            else:
                word = PIECE_WORD.search(text[subject[0]:subject[1]])
                src = _find_piece(board, PIECES[word.group(1).lower()], color_of(text[subject[0]:subject[1]]),
                                  _piece_square(text, subject, mentions)) if word else None
            if src is None or board.piece_at(src) is None:
                fact("attacks", c, None, "subject not found on the board")
                continue
            if re.search(r"defend|protect|guard", claim_text, re.I) and re.search(r"\bun|not|no\b", claim_text, re.I) and not objects:
                ok = not board.attackers(board.color_at(src), src)  # "is undefended"
                fact("attacks", c, ok, "no defenders" if ok else "defended")
                continue
            if re.search(r"pin", claim_text, re.I) and not objects:
                fact("attacks", c, board.is_pinned(board.color_at(src), src) != negated, "pinned")
                continue
            targets = []
            for o in objects:
                if o[2] == "SQUARE":
                    if square_of(text[o[0]:o[1]]) is not None:
                        targets.append(square_of(text[o[0]:o[1]]))
                else:
                    w = PIECE_WORD.search(text[o[0]:o[1]])
                    sq = _find_piece(board, PIECES[w.group(1).lower()], color_of(text[o[0]:o[1]]) if w else None,
                                     _piece_square(text, o, mentions)) if w else None
                    if sq is not None:
                        targets.append(sq)
            if not targets:
                fact("attacks", c, None, "no target")
                continue
            hit = [board.is_attacked_by(board.color_at(src), t) and src in board.attackers(board.color_at(src), t)
                   for t in targets]
            fact("attacks", c, (all(hit) if not negated else not any(hit)),
                 ", ".join(f"{chess.square_name(t)}:{'yes' if h else 'no'}" for t, h in zip(targets, hit)))
        elif kind == "FLIGHT":
            color = None
            if subject is not None and subject[2] == "PIECE":
                color = color_of(text[subject[0]:subject[1]])
                sq = _piece_square(text, subject, mentions)
                if color is None and sq is not None and board.piece_type_at(sq) == chess.KING:
                    color = board.color_at(sq)
            if color is None:
                color = board.turn  # in the frame position, the side to move is the one being mated
            if board.king(color) is None:
                fact("flight", c, None, "no king")
                continue
            reachable = _king_moves(board, color)
            squares = [square_of(text[o[0]:o[1]]) for o in objects if o[2] == "SQUARE"]
            squares = [sq for sq in squares if sq is not None]
            restrictive = negated or bool(RESTRICTIVE.search(claim_text))
            if squares:
                ok = all(s not in reachable for s in squares) if restrictive else all(s in reachable for s in squares)
            else:
                ok = (not reachable) if restrictive else bool(reachable)
            fact("flight", c, ok, "king moves: " + (", ".join(chess.square_name(s) for s in sorted(reachable)) or "none"))

    chosen = next((mv for mv in moves if mv[5] == "CHOSEN" and mv[2] is root), None)
    chosen_uci = chosen[3].uci() if chosen and chosen[3] else None
    summary = {}
    for f in facts:
        s = summary.setdefault(f["kind"], {"true": 0, "false": 0, "unverifiable": 0})
        s["true" if f["verdict"] is True else "false" if f["verdict"] is False else "unverifiable"] += 1
    return {"facts": facts, "summary": summary, "chosen": chosen_uci,
            "chosen_mentioned": chosen is not None,
            "matches_played": (chosen_uci == played) if played and chosen_uci else None}
