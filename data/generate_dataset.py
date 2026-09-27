"""Generate a synthetic dataset for training the chess-move tagger.

Every example is a piece of chess-tutor-style text in which the chess moves and
board locations are wrapped in double brackets. There are six kinds of span:

    [[Qxd5]]     START:    a move played from the current position
    [[>Ne2]]     CONT:     continues the previous move (it is played after it)
    [[@e4]]      SQUARE:   "the pawn on [[@e4]]"
    [[|c-file]]  FILE:     "the [[|c-file]] is open"            (| is a vertical line)
    [[_7th rank]] RANK:    "a rook on the [[_7th rank]]"        (_ is a horizontal line)
    [[/a1-h8]]   DIAGONAL: "the [[/a1-h8]] diagonal"           (/ is a diagonal line)

A single move is just a START with no CONT after it; a series ("line") is a
START followed by CONTs. Clicking a CONT move in the app replays the whole
series up to that move. Example:

    The game continued [[Nb8]] [[>Nf3]] [[>c6]] [[>dxc6]].

`parse_markup` turns that into plain text plus labelled character spans:

    {"text": "The game continued Nb8 Nf3 c6 dxc6.",
     "spans": [[19, 22, "START"], [23, 26, "CONT"], [27, 29, "CONT"], [30, 34, "CONT"]]}

The moves come from real (legal) positions played out with python-chess, and
the templates respect chess logic: a reply always continues the move it
answers, lines always begin from the current position. Squares look exactly
like pawn moves ("the pawn on e4" vs "White played e4"), "file" and "rank"
are also everyday words ("the PGN file", "your rank"), and evaluations,
results and look-alikes (A4, B12) are mixed in *without* brackets: telling
all of these apart from context is the whole point of using a model instead
of a regex. Vague references the app couldn't resolve ("the long diagonal",
"the back rank") are left unlabelled.

Usage:
    python data/generate_dataset.py            # writes data/{train,val,test}.jsonl
    python data/generate_dataset.py --train 2000
"""

import argparse
import json
import random
import re
from pathlib import Path

import chess

DATA_DIR = Path(__file__).parent

# --------------------------------------------------------------------------
# Markup helpers
# --------------------------------------------------------------------------

MARK = re.compile(r"\[\[([>@|_/]?)(.+?)\]\]")
KINDS = {"": "START", ">": "CONT", "@": "SQUARE", "|": "FILE", "_": "RANK", "/": "DIAGONAL"}


def parse_markup(marked: str) -> dict:
    """'played [[e4]] [[>e5]] on [[@e5]]' -> text 'played e4 e5 on e5' + spans [[7, 9, 'START'], [10, 12, 'CONT'], [16, 18, 'SQUARE']]"""
    text, spans, cursor, length = [], [], 0, 0
    for match in MARK.finditer(marked):
        before = marked[cursor:match.start()]
        move = match.group(2)
        start = length + len(before)
        spans.append([start, start + len(move), KINDS[match.group(1)]])
        text += [before, move]
        length = start + len(move)
        cursor = match.end()
    text.append(marked[cursor:])
    return {"text": "".join(text), "spans": spans, "marked": marked}


ANNOTATIONS = ["!", "?", "!?", "?!", "!!", "??"]


def mark(san: str, rng: random.Random, cont: bool = False, decorate: bool = True) -> str:
    """Wrap one move in [[ ]] (or [[> ]] for CONT), sometimes adding things that are NOT part of it."""
    if san.startswith("O-O") and rng.random() < 0.15:
        san = san.replace("O", "0")  # 0-0 is a common (informal) way to write castling
    out = f"[[{'>' if cont else ''}{san}]]"
    if decorate:
        roll = rng.random()
        if roll < 0.08:
            out += rng.choice(ANNOTATIONS)
        elif roll < 0.12:
            out = f"**{out}**"  # LLM answers often come back as markdown
        elif roll < 0.14:
            out = f"`{out}`"
    return out


def move_prefix(board: chess.Board, rng: random.Random) -> str:
    """Move number in front of a move: '14. ' for White, '14...' for Black."""
    if board.turn == chess.WHITE:
        return f"{board.fullmove_number}." + rng.choice([" ", " ", ""])
    return f"{board.fullmove_number}..." + rng.choice(["", "", " "])


def render_line(board: chess.Board, moves: list[chess.Move], rng: random.Random,
                numbered: bool) -> str:
    """A series of moves from `board`: first one is START, the rest CONT (1. e4 e5 2. Nf3)."""
    board = board.copy()
    parts = []
    for i, move in enumerate(moves):
        prefix = ""
        if numbered and (board.turn == chess.WHITE or i == 0):
            prefix = move_prefix(board, rng)
        parts.append(prefix + mark(board.san(move), rng, cont=i > 0, decorate=rng.random() < 0.3))
        board.push(move)
    return " ".join(parts)


# --------------------------------------------------------------------------
# Positions and moves
# --------------------------------------------------------------------------

OPENINGS = [
    "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7",
    "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6",
    "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7",
    "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O",
    "e4 e6 d4 d5 Nc3 Bb4 e5 c5",
    "e4 c6 d4 d5 e5 Bf5 Nf3 e6",
    "e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d4 exd4",
    "d4 d5 Bf4 Nf6 e3 c5 c3 Nc6",
    "e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6",
    "c4 e5 Nc3 Nf6 g3 d5 cxd5 Nxd5",
    "Nf3 d5 g3 Nf6 Bg2 e6 O-O Be7",
    "e4 e5 f4 exf4 Nf3 g5",
    "d4 f5 g3 Nf6 Bg2 g6",
    "e4 g6 d4 Bg7 Nc3 d6 Be3 a6",
    "d4 Nf6 c4 e6 Nc3 Bb4 Qc2 O-O a3 Bxc3+ Qxc3",
    "e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4",
    "e4 c5 Nc3 Nc6 g3 g6 Bg2 Bg7",
    "d4 d5 c4 dxc4 e4 e5 Nf3 exd4 Bxc4 Bb4+",
]

PIECE_NAMES = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop",
               chess.ROOK: "rook", chess.QUEEN: "queen", chess.KING: "king"}


def pick_move(board: chess.Board, rng: random.Random) -> chess.Move:
    """A random legal move, biased towards captures/checks so 'x' and '+' show up often."""
    moves = list(board.legal_moves)
    if rng.random() < 0.35:
        forcing = [m for m in moves if board.is_capture(m) or board.gives_check(m)]
        if forcing:
            return rng.choice(forcing)
    return rng.choice(moves)


def play_random(board: chess.Board, n: int, rng: random.Random) -> list[chess.Move]:
    board = board.copy()
    moves = []
    for _ in range(n):
        if board.is_game_over():
            break
        move = pick_move(board, rng)
        moves.append(move)
        board.push(move)
    return moves


def promotion_position(rng: random.Random) -> chess.Board:
    """A small endgame where the side to move can promote (so e8=Q, bxa1=N... appear)."""
    while True:
        board = chess.Board(None)
        color = rng.choice([chess.WHITE, chess.BLACK])
        rank = 6 if color == chess.WHITE else 1
        board.set_piece_at(chess.square(rng.randrange(8), rank), chess.Piece(chess.PAWN, color))
        for piece_type in rng.sample([chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN], 2):
            sq = rng.randrange(64)
            if board.piece_at(sq) is None and chess.square_rank(sq) not in (0, 7):
                board.set_piece_at(sq, chess.Piece(piece_type, rng.choice([chess.WHITE, chess.BLACK])))
        for king_color in (chess.WHITE, chess.BLACK):
            while True:
                sq = rng.randrange(64)
                if board.piece_at(sq) is None:
                    board.set_piece_at(sq, chess.Piece(chess.KING, king_color))
                    break
        board.turn = color
        board.fullmove_number = rng.randint(35, 70)
        if board.is_valid() and any(m.promotion for m in board.legal_moves):
            return board


def random_position(rng: random.Random) -> chess.Board:
    """The 'current position': the position right before the move being discussed."""
    if rng.random() < 0.08:
        return promotion_position(rng)
    while True:
        board = chess.Board()
        opening = rng.choice(OPENINGS).split()
        for san in opening[:rng.randint(0, len(opening))]:
            board.push_san(san)
        for move in play_random(board, rng.randint(0, 50), rng):
            board.push(move)
        if len(list(board.legal_moves)) > 1 and not board.is_game_over():
            return board


# --------------------------------------------------------------------------
# Context: everything a template might want to mention
# --------------------------------------------------------------------------

FILES = "abcdefgh"


def square(name: str) -> str:
    return f"[[@{name}]]"


def random_square(rng: random.Random) -> str:
    return square(rng.choice(FILES) + str(rng.randint(1, 8)))


def file_ref(rng: random.Random) -> str:
    letter = rng.choice(FILES)
    return f"[[|{letter}-file]]" if rng.random() < 0.85 else f"[[|{letter} file]]"


ORDINALS = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"]
ORDINAL_WORDS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]


def rank_ref(rng: random.Random) -> str:
    if rng.random() < 0.15:
        return "back rank"  # depends on whose side: not labelled
    i = rng.randrange(8)
    return f"[[_{rng.choice([ORDINALS, ORDINAL_WORDS])[i]} rank]]"


# Every diagonal at least 3 squares long, as (start, end) square names.
DIAGONALS = []
for step in (9, 7):  # a1-h8 direction and h1-a8 direction
    for sq in chess.SQUARES:
        prev = sq - step
        if 0 <= prev < 64 and abs(chess.square_file(prev) - chess.square_file(sq)) == 1:
            continue  # not the start of a diagonal
        end = sq
        while 0 <= end + step < 64 and abs(chess.square_file(end + step) - chess.square_file(end)) == 1:
            end += step
        if chess.square_distance(sq, end) >= 2:
            DIAGONALS.append((chess.square_name(sq), chess.square_name(end)))


def diagonal_ref(rng: random.Random) -> str:
    """A whole phrase: 'the a1-h8 diagonal', 'the diagonal h7-b1', or the unlabelled 'the long diagonal'."""
    if rng.random() < 0.15:
        return "the long diagonal"  # which one? not labelled
    a, b = rng.sample(rng.choice(DIAGONALS), 2)
    name = f"[[/{a}{rng.choice(['-', '-', '–'])}{b}]]"
    return f"the {name} diagonal" if rng.random() < 0.8 else f"the diagonal {name}"


def evaluation(rng: random.Random) -> str:
    value = rng.uniform(-4.5, 4.5)
    return f"{value:+.1f}" if rng.random() < 0.8 else f"{value:+.2f}"


def piece_list(board: chess.Board, color: chess.Color, rng: random.Random) -> str:
    squares = [sq for sq in chess.SQUARES
               if board.piece_at(sq) and board.piece_at(sq).color == color]
    if not squares or rng.random() < 0.15:
        return "no pieces"
    chosen = rng.sample(squares, min(len(squares), rng.randint(1, 3)))
    return ", ".join(f"{PIECE_NAMES[board.piece_at(sq).piece_type]} on {square(chess.square_name(sq))}"
                     for sq in chosen)


def line_from(board: chess.Board, first: chess.Move, rng: random.Random, extra: int) -> list[chess.Move]:
    after = board.copy()
    after.push(first)
    return [first] + play_random(after, extra, rng)


def sans(board: chess.Board, moves: list[chess.Move]) -> list[str]:
    board = board.copy()
    out = []
    for move in moves:
        out.append(board.san(move))
        board.push(move)
    return out


def make_context(rng: random.Random) -> dict:
    board = random_position(rng)
    legal = list(board.legal_moves)
    played = pick_move(board, rng)
    others = [m for m in legal if m != played]
    best = rng.choice(others)
    alt = rng.choice([m for m in others if m != best] or others)

    # Lines: always start from the current position.
    played_line = line_from(board, played, rng, rng.randint(2, 7))
    best_line = line_from(board, best, rng, rng.randint(2, 7))
    game_line = line_from(board, played, rng, rng.randint(2, 7))
    played_san = sans(board, played_line)
    best_san = sans(board, best_line)
    # Replies: P1/P2 continue PLAYED, B1/B2 continue BEST. Short lines fall back to plausible strings.
    p1, p2 = (played_san[1:] + ["Kh1", "Qd2"])[:2]
    b1, b2 = (best_san[1:] + ["Kg8", "Re1"])[:2]

    mover = "White" if board.turn == chess.WHITE else "Black"

    # A pawn push and the square it lands on: "the pawn to e5 with e5" is the
    # hardest case — the same string is once a square and once a move.
    pushes = [m for m in legal if board.piece_type_at(m.from_square) == chess.PAWN
              and not board.is_capture(m) and not m.promotion]
    push = rng.choice(pushes) if pushes else None
    piece_moves = [m for m in legal if board.piece_type_at(m.from_square) != chess.PAWN]
    piece_move = rng.choice(piece_moves) if piece_moves else best
    castles = [board.san(m) for m in legal if board.is_castling(m)]
    occupant = board.piece_at(played.to_square)
    after_played = board.copy()
    after_played.push(played)

    return {
        # --- moves from the current position: START ---
        "PLAYED": mark(played_san[0], rng),
        "BEST": mark(best_san[0], rng),
        "ALT": mark(board.san(alt), rng),
        "PUSH": mark(board.san(push) if push else "e4", rng),
        "PIECEMOVE": mark(board.san(piece_move), rng),
        "CASTLE": mark(rng.choice(castles) if castles else rng.choice(["O-O", "O-O-O"]), rng),
        "NPLAYED": move_prefix(board, rng) + mark(played_san[0], rng),
        "NBEST": move_prefix(board, rng) + mark(best_san[0], rng),
        # --- moves that continue the previous move: CONT ---
        "P1": mark(p1, rng, cont=True),
        "P2": mark(p2, rng, cont=True),
        "B1": mark(b1, rng, cont=True),
        "B2": mark(b2, rng, cont=True),
        "MATE": mark(p1.rstrip("+#") + "#", rng, cont=True),
        # --- whole lines: START followed by CONTs ---
        "BESTLINE": render_line(board, best_line, rng, numbered=rng.random() < 0.3),
        "PLAYEDLINE": render_line(board, played_line, rng, numbered=rng.random() < 0.3),
        "GAMELINE": render_line(board, game_line, rng, numbered=rng.random() < 0.2),
        "NUMLINE": render_line(board, rng.choice([played_line, best_line]), rng, numbered=True),
        # --- squares are sq, sq2, push_sq, piece_from, piece_to, played_sq (SQUARE) ---
        # --- everything else is plain text ---
        "mover": mover,
        "opponent": "Black" if mover == "White" else "White",
        "sq": random_square(rng),
        "sq2": random_square(rng),
        "push_sq": square(chess.square_name(push.to_square) if push else "e4"),
        "piece_from": square(chess.square_name(piece_move.from_square)),
        "piece_to": square(chess.square_name(piece_move.to_square)),
        "piece": PIECE_NAMES[board.piece_type_at(piece_move.from_square)],
        "file": file_ref(rng),
        "file2": file_ref(rng),
        "rank": rank_ref(rng),
        "diag": diagonal_ref(rng),
        "ordinal": rng.choice(ORDINALS + ORDINAL_WORDS[:3]),
        "eval": evaluation(rng),
        "eval2": evaluation(rng),
        "loss": f"{rng.uniform(0.3, 9.0):.1f}",
        "ply": str(rng.randint(1, 120)),
        "moveno": str(board.fullmove_number),
        "result": rng.choice(["1-0", "0-1", "1/2-1/2", "½-½"]),
        "rating": str(rng.randint(1100, 2850)),
        "occupant": ("empty" if occupant is None else
                     f"occupied by a {'White' if occupant.color else 'Black'} {PIECE_NAMES[occupant.piece_type]}"),
        "wpieces": piece_list(after_played, chess.WHITE, rng),
        "bpieces": piece_list(after_played, chess.BLACK, rng),
        "played_sq": square(chess.square_name(played.to_square)),
        "classification": rng.choice(["Blunder", "Mistake", "Inaccuracy", "Good", "Best", "Excellent"]),
        "paper": rng.choice(["A4", "A3", "B5", "C4"]),
        "code": rng.choice(["B12", "D3", "E4", "F1", "G7", "H2", "C3PO", "R2-D2", "Q3"]),
    }


# --------------------------------------------------------------------------
# Templates
#
# UPPERCASE placeholders are moves (become labelled spans); lowercase ones are
# plain text, except the square placeholders (sq, sq2, push_sq, piece_from,
# piece_to, played_sq), which become SQUARE spans, and file/file2/rank/diag,
# which become FILE/RANK/DIAGONAL spans (diag is a whole "the ... diagonal" phrase). PLAYED/BEST/ALT/... are START moves; P1, P2 continue PLAYED and
# B1, B2 continue BEST, so a template must only use P1 after PLAYED and B1
# after BEST. TRAIN_TEMPLATES feed train/val; HELDOUT_TEMPLATES only appear in
# the test set, so the test score tells us whether the model learned *moves*
# or just memorised our sentence patterns.
# --------------------------------------------------------------------------

TRAIN_TEMPLATES = [
    # Exact phrasings produced by chess-tutor's backend/server.py
    "Stockfish prefers {mover}'s {BEST} over {PLAYED}. In this short search, {PLAYED} loses about {loss} pawns of evaluation. The game continued {GAMELINE}. Stockfish's line after {BEST} begins {BESTLINE}.",
    "{mover}'s {PLAYED} lets {opponent} play {MATE} on the very next move: checkmate. Stockfish preferred {mover}'s {BEST} instead. That immediate mate is why {PLAYED} is a blunder.",
    "After {mover}'s {PLAYED}, {played_sq} is {occupant}. White controls it with {wpieces}; Black controls it with {bpieces}. The game continued {GAMELINE}.",
    "{PLAYED} · {classification} · Stockfish prefers {BEST}",
    "{mover} played {PLAYED} at ply {ply}. Stockfish preferred {mover}'s {BEST}. A Stockfish continuation after the played move is {PLAYEDLINE}.",
    "Stockfish line after preferred move: {BESTLINE}. Stockfish line after played move: {PLAYEDLINE}.",
    "What actually happened in the game: {GAMELINE}.",
    # Free-form tutor answers (what an LLM tutor tends to write)
    "{PLAYED} was a mistake because it allows {P1}, and {opponent} wins material. {BEST} would have kept the balance.",
    "Instead of {PLAYED}, {mover} should consider {BEST}. The idea is to meet {B1} with {B2}.",
    "The {piece} on {piece_from} wanted to go to {piece_to}, so {PIECEMOVE} was the natural move.",
    "After {NPLAYED}, the strongest reply is {P1}.",
    "Why not {BEST}? Because after {BESTLINE} {opponent} is completely fine.",
    "The engine gives {eval} after {PLAYED}, but {eval2} after {BEST}.",
    "Main line: {NUMLINE}",
    "{mover} could have castled with {CASTLE} instead of playing {PLAYED}.",
    "Pushing the pawn to {push_sq} with {PUSH} gains space, but it weakens {sq}.",
    "{PLAYED} attacks the pawn on {sq}, but {BEST} was more precise.",
    "The bishop on {sq} controls {diag}, so {BEST} challenges it immediately.",
    "Why is {PLAYED} bad here?",
    "What happens if I play {BEST} instead of {PLAYED}?",
    "Can {mover} play {BEST} here, or is {B1} a problem?",
    "The key moment was move {moveno}: {NPLAYED} was an error, and {NBEST} was better.",
    "Good question! After {PLAYED}, {opponent} can respond with {P1}, attacking the {piece} on {sq}.",
    "The {file} is half-open, which makes {BEST} a strong idea.",
    "Stockfish evaluates the position at {eval}. The best continuation is {BESTLINE}.",
    "{BEST} is necessary: after {PLAYED}, {opponent} simply plays {P1}.",
    "If {BEST}, then {B1} is forced, and {B2} wins, because the {rank} is weak.",
    "The move {PLAYED} ({classification}) drops the evaluation from {eval} to {eval2}.",
    "1. The problem with {PLAYED} is {P1}.\n2. {BEST} keeps control of {sq}.\n3. After {B1}, {mover} can continue with {B2}.",
    "- Played: {PLAYED}\n- Best: {BEST}\n- Engine line: {BESTLINE}",
    "In the game {mover} chose {NPLAYED}; the continuation was {GAMELINE}.",
    "{BEST}! is the only move. Anything else, like {PLAYED}, loses to {P1}.",
    "Stockfish recommends {BEST} rather than {PLAYED}. After {BEST}, the likely line is {BESTLINE}.",
    "Both {BEST} and {ALT} are fine here; only {PLAYED} is a mistake.",
    "After {PLAYED} {P1}, {mover} has to find {P2}.",
    "The key continuation Stockfish chose was:\n\n{BEST}\n(which leads to a sharp, balanced game)",
    "The weak squares in {mover}'s camp are {sq} and {sq2}.",
    "Put the {piece} on {sq}, where it controls {sq2}.",
    "{PLAYED} leaves {sq} undefended, so {P1} lands on {sq2} with tempo.",
    "The rook belongs on the {file}, so {BEST} is natural.",
    "{PLAYED} opens the {file} for {opponent}'s rooks.",
    "A rook on the {rank} is very strong here, and {BEST} gets it there.",
    "The bishop on {sq} is aiming down {diag} toward {sq2}.",
    "{mover}'s pieces are tied to the {rank}, and the {file} is completely open.",
    "Doubling rooks on the {file} with {BEST} is the plan.",
    "After {PLAYED}, {diag} is closed for good.",
    "{BEST} brings the rook to the {rank}, where it attacks the pawns.",
    "{PIECEMOVE} puts the bishop on {diag}, pointing straight at the king.",
    "The queen and bishop battery on {diag} is the reason {BEST} works.",
    "After {PLAYED}, the king is stuck on the {rank} and {P1} is coming.",
    "The rooks are connected on the {rank} after {CASTLE}.",
    "Passed pawns on the {rank} are very dangerous, so {BEST} is urgent.",
    "Both rooks on the {rank} and the queen on {sq}: {mover} is crashing through.",
    "The {file} and the {file2} are both open, so {BEST} puts a rook on one of them right away.",
    # No moves, only squares and chess-looking text
    "The {file} is open and White controls the {sq} square.",
    "The game ended {result} after {moveno} moves.",
    "White is slightly better, around {eval}, thanks to the knight outpost on {sq}.",
    "After {moveno} moves the evaluation is {eval}. Both kings are safe and {diag} is closed.",
    "A {piece} on {sq} is usually stronger than one on {sq2}.",
    "Please print the scoresheet on {paper} paper and bring it to board {moveno}.",
    "Player rated {rating} lost {result} in round {ply}.",
    "Which pieces defend {sq}? The {piece} on {sq2} is the only defender.",
    "Room {code} is where the blitz tournament starts at {moveno}:00.",
    "The {file} and the {rank} are under control, and {diag} is wide open.",
    "Download the PGN file and check your rank in the standings: you finished {ordinal}.",
    "Save the analysis to a file; your club rank went up to {ordinal} this month.",
]

HELDOUT_TEMPLATES = [
    "Honestly, {PLAYED} isn't terrible, but {BEST} is cleaner: the point is {BESTLINE}.",
    "The critical try is {BEST}. Then {B1} is forced and {B2} wins on the spot.",
    "In the game, {mover} went for {PLAYED}, and after {P1} the pawn on {sq} fell.",
    "Squares like {sq} and {sq2} are weak, so {BEST} makes a lot of sense.",
    "Line: {NUMLINE} — with a clear edge.",
    "{opponent} answered {PLAYED} with {P1}, and now the {file} is open.",
    "No moves here, just a note: the {file} and the {sq} square matter most.",
    "My engine says {eval}. I would start with {BEST}, and only after {B1} play {B2}.",
    "Moving the {piece} from {piece_from} to {piece_to} ({PIECEMOVE}) was the idea.",
    "You played {PLAYED}. A stronger option was {BEST}, and the engine expected {BESTLINE}.",
    "Tip: after {PUSH}, the pawn on {push_sq} controls {sq}.",
    "Result {result}; the decisive mistake was {NPLAYED}.",
    "Both {ALT} and {BEST} were playable, but {PLAYED} walks into {P1} {P2}.",
    "Control of the {file} decides the game, and {BEST} fights for it.",
    "Rooks on the {rank}: that's where {BEST} is heading.",
    "{PIECEMOVE} puts a piece on {diag}, eyeing {sq}.",
    "You finished {ordinal} in the ranking, but your rook on the {rank} was great.",
]

HANDWRITTEN_FILE = DATA_DIR / "handwritten.txt"


def fill(template: str, ctx: dict) -> str:
    return re.sub(r"\{(\w+)\}", lambda m: ctx[m.group(1)], template)


def generate(templates: list[str], n: int, rng: random.Random, source: str) -> list[dict]:
    examples = []
    for _ in range(n):
        index = rng.randrange(len(templates))
        example = parse_markup(fill(templates[index], make_context(rng)))
        example["source"] = f"{source}:{index}"
        examples.append(example)
    return examples


def load_handwritten() -> list[dict]:
    examples = []
    for line in HANDWRITTEN_FILE.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            example = parse_markup(line.replace("\\n", "\n"))
            example["source"] = "handwritten"
            examples.append(example)
    return examples


def check(example: dict) -> None:
    """Sanity check: clean spans, and a CONT always has a move before it to continue."""
    seen_move = False
    for start, end, label in example["spans"]:
        span = example["text"][start:end]
        assert span == span.strip(" .,;:!?*`()"), f"bad span {span!r} in {example['marked']!r}"
        assert label != "CONT" or seen_move, f"CONT without a move before it: {example['marked']!r}"
        seen_move = seen_move or label != "SQUARE"
    assert "[[" not in example["text"] and "]]" not in example["text"]


def write_jsonl(path: Path, examples: list[dict]) -> None:
    with path.open("w") as f:
        for example in examples:
            check(example)
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    counts = {kind: sum(label == kind for e in examples for *_, label in e["spans"]) for kind in KINDS.values()}
    print(f"wrote {len(examples):5d} examples ({', '.join(f'{n:5d} {k}' for k, n in counts.items())}) -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=int, default=1000)
    parser.add_argument("--val", type=int, default=150)
    parser.add_argument("--test", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    train = generate(TRAIN_TEMPLATES, args.train, rng, "train_template")
    val = generate(TRAIN_TEMPLATES, args.val, rng, "train_template")
    test = generate(HELDOUT_TEMPLATES, args.test, rng, "heldout_template") + load_handwritten()
    rng.shuffle(test)

    write_jsonl(DATA_DIR / "train.jsonl", train)
    write_jsonl(DATA_DIR / "val.jsonl", val)
    write_jsonl(DATA_DIR / "test.jsonl", test)


if __name__ == "__main__":
    main()
