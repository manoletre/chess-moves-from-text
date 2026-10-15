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
FALLBACK = " (checked before the unplayable move)"
REACH = re.compile(r"reach|go to|move to|get to|land on|jump to", re.I)
ANAPHORA = re.compile(r"\s*(that square|this square|those squares|these squares|it\b|them\b|both)", re.I)
SENTENCE_END = re.compile(r"[.;!?\n]")
RECAPTURE = re.compile(r"\b(recaptur\w*|takes? (it|back)|captur\w* (it|back)|retak\w*)\b", re.I)
MATE_WORD = re.compile(r"mate\b", re.I)
CHECK_WORD = re.compile(r"\bcheck", re.I)
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
    move, status, _ = resolve_move(board, text)
    return move, status


def resolve_move(board: chess.Board, text: str, answered: int | None = None):
    """Move text (any notation) -> (move | None, status, note).

    status is 'ok', 'illegal' or 'ambiguous'. `answered` is the square the move being replied to
    landed on, so "recaptures" / "takes it" can be resolved without a square. note is set when the
    move is clear but the text gets part of it wrong ("the bishop on e5 slides to f7" when the bishop
    is on d5): (wrong_square, actual_square)."""
    t = text.strip().replace("…", "").lstrip(".").strip("*`!? ")
    legal = list(board.legal_moves)
    if not legal:
        return None, "illegal", None
    # castling
    m = CASTLE.search(t)
    if m:
        long = (m.group(2) and m.group(2).count("-") == 2) or (m.group(1) and m.group(1).lower().startswith("q"))
        try:
            return board.parse_san("O-O-O" if long else "O-O"), "ok", None
        except ValueError:
            return None, "illegal", None
    # UCI ("c6c6" is not a move at all)
    m = UCI.match(t.rstrip("+#"))
    if m:
        try:
            move = chess.Move.from_uci(t.rstrip("+#"))
        except ValueError:
            return None, "illegal", None
        return (move, "ok", None) if move in legal else (None, "illegal", None)
    # SAN (python-chess ignores a missing/extra x, + or #: those are checked as separate facts)
    if re.fullmatch(r"[KQRBNP]?[a-h]?[1-8]?x?[a-h][1-8](=?[QRBN])?[+#]?|O-O(-O)?[+#]?", t):
        try:
            return board.parse_san(t), "ok", None
        except chess.AmbiguousMoveError:
            return None, "ambiguous", None
        except ValueError:
            return None, "illegal", None
    # long algebraic / arrows / prose: filter the legal moves by every constraint we can read
    candidates = legal
    two = TWO_SQUARES.search(t)
    squares = SQUARE.findall(t)
    from_square = None
    if two:
        from_square = chess.parse_square(two.group(1))
        to = chess.parse_square(two.group(2))
        candidates = [mv for mv in candidates if mv.to_square == to]
    elif squares:
        to = chess.parse_square(squares[-1])
        candidates = [mv for mv in candidates if mv.to_square == to]
        if len(squares) >= 2:
            from_square = chess.parse_square(squares[0])
    elif answered is not None and RECAPTURE.search(t):  # "recaptures", "takes it", "takes back"
        candidates = [mv for mv in candidates if mv.to_square == answered]
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
        return None, "ambiguous", None  # nothing to go on
    # a check or mate said inside the move ("the queen checks", "delivers checkmate on e5") narrows it down
    if len(candidates) > 1 and MATE_WORD.search(t):
        mates = [mv for mv in candidates if _gives_mate(board, mv)]
        candidates = mates or candidates
    elif len(candidates) > 1 and CHECK_WORD.search(t):
        checks = [mv for mv in candidates if board.gives_check(mv)]
        candidates = checks or candidates
    note = None
    if from_square is not None:
        exact = [mv for mv in candidates if mv.from_square == from_square]
        if not exact and len(candidates) == 1:  # the right move from the wrong square: say so, but resolve it
            note = (from_square, candidates[0].from_square)
        else:
            candidates = exact
    if len(candidates) == 1:
        return candidates[0], "ok", note
    return None, "illegal" if not candidates else "ambiguous", None


def _gives_mate(board, move):
    b = board.copy(stack=False)
    b.push(move)
    return b.is_checkmate()


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


def _line_squares(text: str, kind: str) -> list:
    """The squares of a FILE / RANK / DIAGONAL mention ("c-file", "7th rank", "a1-h8")."""
    t = text.lower()
    if kind == "FILE":
        f = t.strip()[0]
        return [chess.square(chess.FILE_NAMES.index(f), r) for r in range(8)] if f in chess.FILE_NAMES else []
    if kind == "RANK":
        digit = re.search(r"[1-8]", t)
        words = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]
        rank = int(digit.group()) if digit else next((i + 1 for i, w in enumerate(words) if w in t), None)
        return [chess.square(f, rank - 1) for f in range(8)] if rank else []
    ends = re.findall(r"[a-h][1-8]", t)
    if len(ends) != 2:
        return []
    a, b = (chess.parse_square(x) for x in ends)
    if abs(chess.square_file(a) - chess.square_file(b)) != abs(chess.square_rank(a) - chess.square_rank(b)):
        return []  # not a diagonal
    return list(chess.SquareSet.between(a, b)) + [a, b]


def _piece_line(text: str, mention, mentions):
    """'bishop sits on the a4-e8 diagonal', 'a rook on the e-file' -> the line mention after the piece."""
    for m in mentions:
        if m[2] in ("FILE", "RANK", "DIAGONAL") and m[0] >= mention[1] and \
                re.fullmatch(r"\s*(?:\w+\s+){0,2}(?:on|along|down|up)\s+the\s*", text[mention[1]:m[0]]):
            return m
    return None


def _moved_from(last_move, square, board, kind=None):
    """The piece that just moved, named by the square it came from ("Rxh7 ... the rook on g7 covers h8"):
    its new square, or None."""
    if last_move is None or last_move[3] is None or square is None or last_move[3].from_square != square:
        return None
    to = last_move[3].to_square
    return to if kind is None or board.piece_type_at(to) == kind else None


def _find_piece(board, kind, color, square=None):
    if square is not None:
        return square
    squares = [sq for sq in chess.SQUARES if board.piece_type_at(sq) == kind
               and (color is None or board.color_at(sq) == color)]
    return squares[0] if len(squares) == 1 else None


def _on_line(board, square, text, kind) -> bool:
    """Is a piece that moves along this line standing on it? ("the rook controls the e-file")"""
    piece = board.piece_type_at(square)
    t = text.lower()
    if kind == "FILE":
        f = t.strip()[0]
        return piece in (chess.ROOK, chess.QUEEN) and chess.FILE_NAMES[chess.square_file(square)] == f
    if kind == "RANK":
        digit = re.search(r"[1-8]", t)
        words = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]
        rank = int(digit.group()) if digit else next((i + 1 for i, w in enumerate(words) if w in t), None)
        if rank is None:
            return False
        return piece in (chess.ROOK, chess.QUEEN) and chess.square_rank(square) + 1 == rank
    ends = re.findall(r"[a-h][1-8]", t)
    if len(ends) != 2 or piece not in (chess.BISHOP, chess.QUEEN):
        return False
    return square in _line_squares(text, "DIAGONAL")


def _king_moves(board: chess.Board, color) -> set:
    b = board.copy(stack=False)
    if b.turn != color:
        b.turn = color
        b.ep_square = None
    king = b.king(color)
    return {mv.to_square for mv in b.legal_moves if mv.from_square == king}


def verify(fen: str, text: str, tagged: dict, played: str | None = None) -> dict:
    root = chess.Board(fen)
    defender = not root.turn  # the side the puzzle is played against
    mentions = sorted(tagged["mentions"], key=lambda m: m[0])
    claims = sorted(tagged.get("claims") or [], key=lambda c: c[0])
    facts, moves = [], []  # moves: (start, end, before, move, after, stance)

    state = {"fallback": False}  # set while checking in the position before an unplayable move

    def fact(kind, span, verdict, detail=""):
        facts.append({"kind": kind, "start": span[0], "end": span[1], "text": text[span[0]:span[1]],
                      "verdict": verdict, "detail": detail + (FALLBACK if state["fallback"] else ""),
                      "fallback": state["fallback"]})

    # 1. replay the lines
    prev = None  # (before, after, move, answered_square) of the previous move
    for m in mentions:
        if m[2] != "MOVE":
            continue
        line = m[3] if prev else "NEW"
        before = root if line == "NEW" else (prev[1] if line == "NEXT" else prev[0])
        # the square of the move this one answers: "recaptures" happens there
        answered = None if line == "NEW" else (prev[2].to_square if line == "NEXT" and prev[2] else prev[3])
        move, status, after, note = None, "unknown position", None, None
        if before is not None:
            move, status, note = resolve_move(before, text[m[0]:m[1]], answered)
            if move is not None:
                after = before.copy(stack=False)
                after.push(move)
        moves.append((m[0], m[1], before, move, after, m[4]))
        prev = (before, after, move, answered)
        if status == "ok":
            fact("legal_move", m, True, move.uci())
            if note:
                fact("placement", m, False, f"not on {chess.square_name(note[0])}: the piece is on {chess.square_name(note[1])}")
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
        """(board, last move, fallback). After a move that can't be played (hallucinated, or a reply to
        one), the claim is checked in the last real position before it, and fallback says so."""
        if frame == "NOW":
            return root, None, False
        earlier = [mv for mv in moves if mv[1] <= span[0]]
        if not earlier:
            return root, None, False
        last = earlier[-1]
        if last[4] is not None:
            return last[4], last, False
        for mv in reversed(earlier):  # walk back to a position that exists
            if mv[4] is not None and mv[1] <= last[0]:
                return mv[4], mv, True
        return root, None, True

    # 2. piece placements ("queen on h7", "the g2 pawn", "the bishop on the a4-e8 diagonal")
    for m in mentions:
        if m[2] != "PIECE":
            continue
        square = _piece_square(text, m, mentions)
        sq_mention = next((s for s in mentions if s[2] == "SQUARE" and s[0] >= m[1]
                           and re.fullmatch(r"\s*(on|at|from|in)\s*", text[m[1]:s[0]])), None)
        if square is None:  # "the g2 pawn": the square written before the piece
            sq_mention = next((s for s in mentions if s[2] == "SQUARE" and s[1] <= m[0]
                               and re.fullmatch(r"\s*", text[s[1]:m[0]])), None)
            square = square_of(text[sq_mention[0]:sq_mention[1]]) if sq_mention else None
        span = (min(m[0], sq_mention[0]), max(m[1], sq_mention[1])) if square is not None and sq_mention else m
        if square is None:
            line = _piece_line(text, m, mentions)
            word = PIECE_WORD.search(text[m[0]:m[1]])
            if line is None or word is None:
                continue
            board, last, fallback = frame_board(m, m[5])
            squares = _line_squares(text[line[0]:line[1]], line[2])
            kind, color = PIECES[word.group(1).lower()], color_of(text[m[0]:m[1]])
            found = [sq for sq in squares if board.piece_type_at(sq) == kind and (color is None or board.color_at(sq) == color)]
            state["fallback"] = fallback
            fact("placement", (m[0], line[1]), bool(found), (f"on {chess.square_name(found[0])}" if found else "none") +
                 f" on the {text[line[0]:line[1]]}")
            state["fallback"] = False
            continue
        board, last, fallback = frame_board(m, m[5])
        word = PIECE_WORD.search(text[m[0]:m[1]])
        if word is None:
            continue
        kind, color = PIECES[word.group(1).lower()], color_of(text[m[0]:m[1]])
        moved = _moved_from(last, square, board, kind)
        state["fallback"] = fallback
        if moved is not None:  # the piece that just moved, called by its old square: not a false placement
            fact("placement", span, True, f"{chess.square_name(square)}: the {chess.piece_name(kind)} that just moved there")
        else:
            ok = board.piece_type_at(square) == kind and (color is None or board.color_at(square) == color)
            fact("placement", span, ok, f"{chess.square_name(square)}: {board.piece_at(square) or 'empty'}")
        state["fallback"] = False

    # 3. claims
    for c in claims:
        start, end, kind, frame = c[0], c[1], c[2], (c[3] if len(c) > 3 else None)
        board, last_move, fallback = frame_board(c, frame)
        state["fallback"] = fallback
        claim_text = text[start:end]
        negated = bool(NEGATION.search(claim_text))
        lo, hi = _clause_bounds(text, start)
        # objects stop where the next claim's subject starts ("covers g1 and h2, and the queen covers g3")
        passive = bool(re.match(r"\s*by\b", text[end:]))
        later = next((x for x in claims if x[0] >= end and x[0] < hi), None)
        if later is not None and not passive:
            next_subject = next((m for m in reversed(mentions) if end <= m[0] < later[0] and m[2] in ("MOVE", "PIECE")), None)
            if next_subject is not None:
                hi = next_subject[0]
        subject = next((m for m in reversed(mentions) if lo <= m[0] < start and m[2] in ("MOVE", "PIECE")), None)
        objects = [m for m in mentions if end <= m[0] < hi and m[2] in ("SQUARE", "PIECE", "FILE", "RANK", "DIAGONAL")]
        if passive:  # "g1 is covered by the rook on g2": the agent is the subject, the thing before it the object
            agent = next((m for m in mentions if end <= m[0] < (later[0] if later else hi) and m[2] == "PIECE"), None)
            if agent is not None:
                covered = next((m for m in reversed(mentions) if lo <= m[0] < start and m[2] in ("SQUARE", "PIECE")), None)
                objects = [covered] if covered is not None else []
                subject = agent

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
                named = _piece_square(text, subject, mentions)
                src = _moved_from(last_move, named, board, PIECES[word.group(1).lower()] if word else None)
                if src is None:
                    src = _find_piece(board, PIECES[word.group(1).lower()], color_of(text[subject[0]:subject[1]]),
                                      named) if word else None
            if src is None or board.piece_at(src) is None:
                named = _piece_square(text, subject, mentions) if subject[2] == "PIECE" else None
                if named is not None:  # "the rook on c7 covers ..." with no rook on c7: false, not unknowable
                    fact("attacks", c, False, f"no such piece on {chess.square_name(named)}")
                else:
                    fact("attacks", c, None, "subject not found on the board")
                continue
            if re.search(r"defend|protect|guard", claim_text, re.I) and re.search(r"\bun|not|no\b", claim_text, re.I) and not objects:
                ok = not board.attackers(board.color_at(src), src)  # "is undefended"
                fact("attacks", c, ok, "no defenders" if ok else "defended")
                continue
            if re.search(r"pin", claim_text, re.I):  # "pinned (to the king by the knight)": objects only say by what
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
            lines = [o for o in objects if o[2] in ("FILE", "RANK", "DIAGONAL")]
            if not targets and lines:  # "controls the e-file": a line piece standing on that line
                ok = all(_on_line(board, src, text[o[0]:o[1]], o[2]) for o in lines)
                fact("attacks", c, ok != negated, ("on " if ok else "not on ") + ", ".join(text[o[0]:o[1]] for o in lines))
                continue
            if not targets and ANAPHORA.match(text[end:]):  # "covers that square / it / both": the square(s) named before
                sentence_start = max([m.end() for m in SENTENCE_END.finditer(text) if m.end() <= start] or [0])
                earlier = [square_of(text[m[0]:m[1]]) for m in mentions
                           if m[2] == "SQUARE" and sentence_start <= m[0] < subject[0]]
                earlier = [sq for sq in earlier if sq is not None and sq != src]
                take = 2 if re.match(r"\s*(both|them|those|these)", text[end:], re.I) else 1
                targets = earlier[-take:]
            targets = list(dict.fromkeys(targets))
            if not targets:
                fact("attacks", c, None, "no target")
                continue
            own = board.color_at(src)
            hit = [src in board.attackers(own, t) for t in targets]
            if REACH.search(claim_text):  # "cannot reach e7": a square its own piece stands on can't be reached
                hit = [h and board.color_at(t) != own for h, t in zip(hit, targets)]
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
                color = defender  # a king with no colour is the one being attacked
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

    state["fallback"] = False
    # the reflection's conclusion: the last chosen move that can be played from the puzzle position
    # (a reply mistaken for a new chosen move can't be played there, so it doesn't override the decision)
    chosen_all = [mv for mv in moves if mv[5] == "CHOSEN" and mv[2] is root]
    chosen = next((mv for mv in reversed(chosen_all) if mv[3] is not None), chosen_all[-1] if chosen_all else None)
    chosen_uci = chosen[3].uci() if chosen and chosen[3] else None
    summary = {}
    for f in facts:
        s = summary.setdefault(f["kind"], {"true": 0, "false": 0, "unverifiable": 0})
        s["true" if f["verdict"] is True else "false" if f["verdict"] is False else "unverifiable"] += 1
    return {"facts": facts, "summary": summary, "chosen": chosen_uci,
            "chosen_mentioned": chosen is not None,
            "matches_played": (chosen_uci == played) if played and chosen_uci else None}
