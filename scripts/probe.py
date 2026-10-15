"""Probe the tagger with phrasings it may not have seen, labelled by construction.

Real puzzle positions + legal moves (python-chess) are rendered into short reflections from
phrase lists: moves in every notation and many prose forms, rejected/chosen wording, and many
ways to say check, mate, wins, attacks and flight. Every phrase list is split in two by a fixed
hash: one half can be used for training (--split train), the other only for probing
(--split probe), so the probe measures generalisation to wording the model never saw.

    python scripts/probe.py --split probe --n 400 --out data/probe/probe.jsonl
    python scripts/probe.py --split train --n 300 --out data/reflections/synthetic/templated.jsonl
    python scripts/probe.py --eval data/probe/probe.jsonl      # recall per phrase family
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from chess_tagger.markup import parse  # noqa: E402

NAMES = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop", chess.ROOK: "rook",
         chess.QUEEN: "queen", chess.KING: "king"}


def half(phrase: str) -> str:
    """Deterministic train/probe split of a phrase."""
    return "train" if int(hashlib.md5(phrase.encode()).hexdigest(), 16) % 2 == 0 else "probe"


# ---------------------------------------------------------------- phrases
# {P} piece, {S} destination, {F} from-square, {V} victim piece. The span of a prose move is
# marked with ⟦ ⟧ inside the template: everything between is the MOVE span.
PROSE_MOVES = [
    "the ⟦{P} goes to {S}⟧", "⟦{P} to {S}⟧", "moving the ⟦{P} to {S}⟧", "the ⟦{P} slides to {S}⟧",
    "the ⟦{P} jumps to {S}⟧", "the ⟦{P} heads for {S}⟧", "bringing the ⟦{P} to {S}⟧", "the ⟦{P} lands on {S}⟧",
    "the ⟦{P} on {F} moves to {S}⟧", "the ⟦{P} from {F} to {S}⟧", "the ⟦{P} steps to {S}⟧",
    "the ⟦{P} swings over to {S}⟧", "putting the ⟦{P} on {S}⟧", "the ⟦{P} comes to {S}⟧",
    "the ⟦{P} drops back to {S}⟧", "the ⟦{P} advances to {S}⟧", "relocating the ⟦{P} to {S}⟧",
    "the ⟦{P} is played to {S}⟧", "the ⟦{P} reroutes to {S}⟧", "sending the ⟦{P} to {S}⟧",
]
PROSE_CAPTURES = [
    "the ⟦{P} takes the {V} on {S}⟧", "⟦capturing the {V} on {S}⟧", "the ⟦{P} captures on {S}⟧",
    "⟦taking on {S} with the {P}⟧", "the ⟦{P} grabs the {V} on {S}⟧", "the ⟦{P} snaps off the {V} on {S}⟧",
    "the ⟦{P} eats the {V} on {S}⟧", "⟦{P} takes {S}⟧", "the ⟦{P} wins the {V} on {S}⟧ outright",
    "⟦removing the {V} on {S}⟧ with the {P}",
]
NOTATIONS = ["san", "san_plain", "uci", "long", "arrow"]
REJECT = [
    "{M} looks tempting, but it fails.", "{M} does not work here.", "I considered {M}, but it is too slow.",
    "{M} is a mistake.", "The natural {M} runs into trouble.", "{M} would be an error.",
    "Not {M}, which lets the opponent off the hook.", "One idea is {M}, but it misses the point.",
    "{M}? That throws away the advantage.", "Avoid {M}, it loses time.", "{M} is refuted easily.",
    "{M} seems attractive, yet it falls short.", "I reject {M}.", "{M} is not good enough.",
]
CHOOSE = [
    "So the move is {M}.", "The best move is {M}.", "I play {M}.", "My choice is {M}.", "{M} is the answer.",
    "The winning move is {M}.", "Therefore {M}.", "I will go with {M}.", "The strongest continuation is {M}.",
    "{M}!", "The key move is {M}.", "Final answer: {M}.", "Best is {M}.", "Let's play {M}.",
]
CHECK = ["<gives check>", "<with check>", "<checks> the {K}", "<is check>", "<puts> the {K} <in check>",
         "<delivers a check>", "<comes with check>", "<checks>", "<giving check>", "<is a check>",
         "<gives a check>", "<forces the king to move>", "<is with check>", "<checks> the enemy {K}"]
MATE = ["<finishes it>", "<it's over>", "<game over>", "<the king is dead>", "<and the king falls>", "<Black resigns>",
        "<forces resignation>", "<is 1-0>", "<is #>", "<no reply exists>", "<can't parry it>", "<there is nothing to be done>",
        "<seals the king's fate>", "<is curtains>", "<lights out>", "<and that's the game>", "<wins on the spot>",
        "<leaves no defence>", "<the game is over>", "<the end>",
        "<delivers checkmate>", "<is mate>", "<mates>", "<is checkmate>", "<ends the game>", "<is mate in one>",
        "<finishes the game>", "<forces mate>", "<leaves> the {K} <no legal move>", "<is a checkmate>", "<mates immediately>",
        "<seals the mate>", "<wins by checkmate>", "<is game over>", "<checkmates>", "<is a mating blow>",
        "<delivers mate>", "<gives mate>"]
WINS = ["<wins> the {V}", "<picks up> the {V}", "<nets> the {V}", "<collects> the {V}", "<wins material>",
        "<grabs> the {V}", "<captures> the {V}", "<wins> a {V}", "<takes> the {V}", "<is winning material>"]
ATTACKS = ["<attacks> {S}", "<covers> {S}", "<controls> {S}", "<eyes> {S}", "<hits> {S}", "<guards> {S}",
           "<defends> {S}", "<protects> {S}", "<watches> {S}", "<targets> {S}", "<x-rays> {S}", "<sweeps> {S}",
           "<keeps an eye on> {S}", "<takes away> {S}", "<sees> {S}", "<supports> {S}"]
NO_ATTACK = ["<does not cover> {S}", "<cannot reach> {S}", "<does not attack> {S}", "<no longer controls> {S}",
             "<fails to guard> {S}"]
EVAL_GOOD = ["<is decisive>", "<is the winning move>", "<is the strongest move>", "<is the only move>", "<is crushing>",
             "<is unstoppable>", "<is lethal>", "<is devastating>", "<is the killer blow>", "<wins by force>",
             "<is the best move>", "<is the right idea>", "<is the correct move>", "<is winning>", "<is overwhelming>",
             "<is the cleanest win>", "<settles it>", "<is a forced win>", "<is completely winning>", "<is the key idea>"]
EVAL_BAD = ["<is a mistake>", "<is a blunder>", "<falls short>", "<is refuted>", "<is too slow>", "<is not the best>",
            "<loses>", "<is dubious>", "<is an error>", "<doesn't win>", "<is not decisive>", "<throws away the win>",
            "<is a dud>", "<goes nowhere>"]
FLIGHT_NONE = ["<has no escape>", "<is trapped>", "<has nowhere to go>", "<cannot move>", "<has no flight squares>",
               "<is boxed in>", "<is stuck>", "<has no legal moves left>", "<is out of squares>", "<cannot run>"]
FLIGHT_SQ = ["<cannot go to> {S}", "<cannot escape to> {S}", "<is denied> {S}", "<cannot step to> {S}",
             "<can't reach> {S}", "<may not use> {S}", "<cannot flee to> {S}", "<is kept off> {S}",
             "<can no longer use> {S}", "<cannot run to> {S}"]
FLIGHT_CAN = ["<can escape to> {S}", "<can run to> {S}", "<can still use> {S}", "<could flee to> {S}",
              "<can step to> {S}", "<can hide on> {S}", "<can still reach> {S}", "<may escape to> {S}",
              "<still has> {S}", "<can slip to> {S}"]
PLACEMENT = ["the {C} {P} on {S}", "{C}'s {P} on {S}", "a {P} on {S}", "the {P} sitting on {S}", "the {P} at {S}"]


def pick(rng, options, split):
    pool = [o for o in options if split == "any" or half(o) == split] or options
    return rng.choice(pool)


def mark_move(board, move, rng, split, line="new", stance="chosen"):
    """One move as markup, in a random notation or prose."""
    kind = rng.choice(NOTATIONS + ["prose"] * 3)
    piece = board.piece_type_at(move.from_square)
    if kind == "san":
        text = board.san(move)
    elif kind == "san_plain":
        text = board.san(move).rstrip("+#")
    elif kind == "uci":
        text = move.uci()
    elif kind == "long":
        text = ("" if piece == chess.PAWN else board.san(move)[0].upper()) + chess.square_name(move.from_square) + \
            ("x" if board.is_capture(move) else "-") + chess.square_name(move.to_square)
    elif kind == "arrow":
        text = chess.square_name(move.from_square) + "→" + chess.square_name(move.to_square)
    else:
        victim = board.piece_type_at(move.to_square)
        if victim and piece != chess.KING:
            template = pick(rng, PROSE_CAPTURES, split)
        else:
            template = pick(rng, PROSE_MOVES, split)
        filled = template.format(P=NAMES[piece], S=chess.square_name(move.to_square),
                                 F=chess.square_name(move.from_square), V=NAMES.get(victim, "piece"))
        before, rest = filled.split("⟦")
        inner, after = rest.split("⟧")
        return f"{before}{{{inner}}}[move {line} {stance}]{after}", "prose:" + template
    return f"{{{text}}}[move {line} {stance}]", kind


def claim(template, frame, **slots):
    """'<covers> {S}' -> '<covers>[attacks after] {e4}[square]' style markup; kind added by caller."""
    return template, frame, slots


def render_claim(template, kind, frame, slots):
    out = template.replace("{K}", "{king}[piece " + frame + "]")
    out = out.replace("<", "\x00").replace(">", f"\x01")
    out = out.replace("{S}", "{" + slots.get("S", "") + "}[square]") if "S" in slots else out
    out = out.replace("{V}", "{" + slots.get("V", "") + "}[piece " + frame + "]") if "V" in slots else out
    out = out.replace("\x00", "<").replace("\x01", f">[{kind} {frame}]")
    return out


def mate_sentence(claim_markup, them):
    """'It <mates>.', 'Then <it's over>.', 'Black <can't parry it>.' depending on how the phrase starts."""
    head = claim_markup.lstrip("<").split()[0].lower()
    if head in {"can't", "cannot"}:
        return f"{them} {claim_markup}."
    if head in {"it's", "game", "the", "and", "black", "no", "there", "lights"}:
        return f"Then {claim_markup}."
    return f"It {claim_markup}."


def example(fen, rng, split):
    board = chess.Board(fen)
    legal = list(board.legal_moves)
    if len(legal) < 3:
        return None
    side = "White" if board.turn else "Black"
    them = "Black" if board.turn else "White"
    families = []
    parts = []
    # 1. a placement now
    own = [sq for sq in chess.SQUARES if board.color_at(sq) == board.turn and board.piece_type_at(sq) != chess.KING]
    if own:
        sq = rng.choice(own)
        t = pick(rng, PLACEMENT, split)
        filled = t.replace("{C}", side).replace("{P}", "\x02").replace("{S}", "{" + chess.square_name(sq) + "}[square]")
        piece_word = NAMES[board.piece_type_at(sq)]
        span = (f"{side}'s {piece_word}" if "{C}'s" in t else f"{side} {piece_word}" if "{C}" in t else piece_word)
        filled = t.replace("{S}", "\x03")
        filled = filled.replace("{C}'s {P}" if "{C}'s" in t else "{C} {P}" if "{C}" in t else "{P}", "{" + span + "}[piece now]")
        filled = filled.replace("\x03", "{" + chess.square_name(sq) + "}[square]")
        parts.append(f"{side} to move. {filled[0].upper()}{filled[1:]} is ready.")
        families.append("placement:" + t)
    # 2. a rejected move with a reply
    rejected = rng.choice(legal)
    m, fam = mark_move(board, rejected, rng, split, "new", "rejected")
    families.append("move:" + fam)
    sentence = pick(rng, REJECT, split)
    families.append("reject:" + sentence)
    after = board.copy()
    after.push(rejected)
    if after.is_check():
        t = pick(rng, CHECK, split)
        families.append("check:" + t)
        sentence = sentence.replace("{M}", "{M} " + render_claim(t, "check", "after", {}), 1)
    parts.append(sentence.replace("{M}", m, 1).replace("{M}", m.split("}")[0].lstrip("{") if False else "it"))
    # 3. the chosen move, with claims about the position after it
    good = [mv for mv in legal if mv != rejected]
    def mates(mv):
        b = board.copy(stack=False)
        b.push(mv)
        return b.is_checkmate()
    good.sort(key=lambda mv: (not mates(mv), not board.gives_check(mv), not board.is_capture(mv)))
    chosen = good[0] if rng.random() < 0.7 else rng.choice(good)
    m, fam = mark_move(board, chosen, rng, split, "new", "chosen")
    families.append("move:" + fam)
    sentence = pick(rng, CHOOSE, split)
    families.append("choose:" + sentence)
    parts.append(sentence.replace("{M}", m))
    after = board.copy()
    after.push(chosen)
    piece = NAMES[after.piece_type_at(chosen.to_square)]
    extra = []
    # claims are labelled whether true or not: also write fake mates / checks (the RL risk)
    if after.is_checkmate() or rng.random() < 0.35:
        t = pick(rng, MATE, split); families.append("mate:" + t)
        extra.append(mate_sentence(render_claim(t, 'mate', 'after', {}), them))
    elif after.is_check() or rng.random() < 0.2:
        t = pick(rng, CHECK, split); families.append("check:" + t)
        extra.append(f"It {render_claim(t, 'check', 'after', {})}.")
    if board.is_capture(chosen) and board.piece_type_at(chosen.to_square):
        t = pick(rng, WINS, split); families.append("wins:" + t)
        extra.append(f"This {render_claim(t, 'wins', 'after', {'V': NAMES[board.piece_type_at(chosen.to_square)]})}.")
    attacked = [sq for sq in after.attacks(chosen.to_square)]
    if attacked:
        t = pick(rng, ATTACKS, split); families.append("attacks:" + t)
        extra.append(f"Now the {{{piece}}}[piece after] {render_claim(t, 'attacks', 'after', {'S': chess.square_name(rng.choice(attacked))})}.")
    not_attacked = [sq for sq in chess.SQUARES if sq not in attacked and sq != chosen.to_square]
    if rng.random() < 0.4:
        t = pick(rng, NO_ATTACK, split); families.append("no_attack:" + t)
        extra.append(f"The {{{piece}}}[piece after] {render_claim(t, 'attacks', 'after', {'S': chess.square_name(rng.choice(not_attacked))})}.")
    king_sq = after.king(not board.turn)
    if king_sq is not None:
        r = rng.random()
        if r < 0.33:
            t = pick(rng, FLIGHT_NONE, split); families.append("flight:" + t)
            extra.append(f"{{{them}'s king}}[piece after] {render_claim(t, 'flight', 'after', {})}.")
        elif r < 0.66:
            t = pick(rng, FLIGHT_SQ, split); families.append("flight:" + t)
            sq = rng.choice(list(chess.SquareSet(chess.BB_KING_ATTACKS[king_sq])))
            extra.append(f"The {{king}}[piece after] {render_claim(t, 'flight', 'after', {'S': chess.square_name(sq)})}.")
        else:
            t = pick(rng, FLIGHT_CAN, split); families.append("flight:" + t)
            sq = rng.choice(list(chess.SquareSet(chess.BB_KING_ATTACKS[king_sq])))
            extra.append(f"The {{king}}[piece after] {render_claim(t, 'flight', 'after', {'S': chess.square_name(sq)})}.")
    if rng.random() < 0.6:
        t = pick(rng, EVAL_GOOD, split); families.append("eval_good:" + t)
        extra.append(f"It {render_claim(t, 'eval', 'after', {})}.")
    if rng.random() < 0.5:
        t = pick(rng, EVAL_BAD, split); families.append("eval_bad:" + t)
        parts[1] = parts[1].rstrip(".") + f", and it {render_claim(t, 'eval', 'after', {})}."
    if rng.random() < 0.3:
        side_lost = them
        t = pick(rng, ["<is lost>", "<is busted>", "<is hopeless>", "<is done for>", "<cannot hold>"], split)
        families.append("eval_side:" + t)
        extra.append(f"{side_lost} {render_claim(t, 'eval', 'after', {})}.")
    rng.shuffle(extra)
    marked = " ".join(parts[:2] + extra[:3] + parts[2:]) if rng.random() < 0.5 else " ".join(parts + extra[:3])
    # if the chosen sentence comes last, claims about it come before it: then they are about the position NOW
    # ... which is not what we wrote. Keep claims after the chosen move.
    marked = " ".join(parts + extra[:3])
    parse(marked)  # must be valid
    return {"marked": marked, "fen": fen, "families": families}


def generate(args):
    rng = random.Random(args.seed)
    fens = [json.loads(l)["fen"] for l in (ROOT / "data/reflections/synthetic_positions.jsonl").read_text().splitlines()]
    raw = [json.loads(l) for l in (ROOT / "data/reflections/raw.jsonl").read_text().splitlines()]
    # training text only from training puzzles, so no test/val position leaks in
    fens += [r["fen"] for r in raw if args.split != "train" or r.get("split") == "train"]
    rng.shuffle(fens)
    out = []
    for fen in fens:
        e = example(fen, rng, args.split)
        if e:
            e["id"] = f"{'p' if args.split == 'probe' else 't'}{len(out):04d}"
            e["category"] = "templated" if args.split == "train" else "probe"
            out.append(e)
        if len(out) == args.n:
            break
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("".join(json.dumps(e) + "\n" for e in out))
    print(f"{len(out)} examples -> {args.out}")


def evaluate(args):
    from chess_tagger.model import ChessTagger
    rows = [json.loads(l) for l in Path(args.eval).read_text().splitlines()]
    tagger = ChessTagger(args.model)
    parsed = [parse(r["marked"]) for r in rows]
    pred = tagger.predict([p["text"] for p in parsed])
    hit, total = Counter(), Counter()
    misses = defaultdict(list)
    for r, g, p in zip(rows, parsed, pred):
        pm = {tuple(m[:5]) for m in p["mentions"]}
        pc = {tuple(c[:3]) for c in p["claims"]}
        for m in g["mentions"]:
            fam = "MOVE" if m[2] == "MOVE" else m[2]
            ok = tuple(m[:5]) in pm
            total[fam] += 1; hit[fam] += ok
        for c in g["claims"]:
            ok = tuple(c[:3]) in pc
            total["claim:" + c[2]] += 1; hit["claim:" + c[2]] += ok
        # per phrase family: did every span of this example come out right?
        all_ok = {tuple(m[:5]) for m in g["mentions"]} <= pm and {tuple(c[:3]) for c in g["claims"]} <= pc
        for f in r["families"]:
            total["fam " + f] += 1; hit["fam " + f] += all_ok
            if not all_ok:
                misses[f].append((r["marked"], p))
    print("recall (span + type + attributes):")
    for k in sorted(t for t in total if not t.startswith("fam ")):
        print(f"  {k:18s} {hit[k] / total[k]:.3f}  ({total[k]})")
    print("\nweakest phrase families (share of examples fully right):")
    fams = sorted((hit[k] / total[k], k, total[k]) for k in total if k.startswith("fam ") and total[k] >= 3)
    for score, k, n in fams[:args.show]:
        print(f"  {score:.2f}  {k[4:]:55s} ({n})")
    if args.out:
        Path(args.out).write_text(json.dumps({k: hit[k] / total[k] for k in total}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "probe", "any"], default="probe")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--eval")
    ap.add_argument("--model", default=str(ROOT / "chess-move-tagger"))
    ap.add_argument("--show", type=int, default=30)
    a = ap.parse_args()
    evaluate(a) if a.eval else generate(a)
