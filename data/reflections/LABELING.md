# Labelling guide (v3 markup)

Each example is a reflection a chess model wrote before playing a move in a puzzle,
plus the position (FEN, side to move = the side that writes the reflection). You copy
the text **character for character** and wrap things in markup. Never fix typos,
spacing, wrong chess, or a text cut off mid-word: label what is written.

The labels are used to *check* the reflection later with python-chess, so label
what the text **claims**, even when the claim is false (an illegal move is still a
move, a wrong "checkmate" is still a mate claim).

## 1. Mentions: `{text}[type]`

| type | what | examples |
|---|---|---|
| `move` | one specific move, in any notation | `{Qxd5}`, `{e8=Q+}`, `{f8h8}` (UCI), `{0-0}`, prose: `{queen slides to h8}`, `{rook from e1 to f8}`, `{capture the pawn on d7}`, `{promote to a knight}`, `{king escapes to d2}`, White can `{recapture the queen}` |
| `piece` | a piece named in words, with its colour if attached | `{queen}`, `{Black's king}`, `{white rook}`, `{knights}`, `{pawns}` |
| `square` | a square | `{e4}` |
| `file` | a file | `{c-file}`, `{h-file}`, `{the e file}`→ just `{e file}` |
| `rank` | a rank | `{7th rank}`, `{eighth rank}`, `{rank 1}` |
| `diagonal` | a named diagonal | `{a1-h8}` diagonal, `{d1-e3 diagonal}`→ just `{d1-e3}` |

A `move` always has two more attributes: `{Qxc3+}[move new rejected]`.

**line**: `new` = played from the current (puzzle) position by the side to move.
`next` = played after the move mentioned just before it, continuing that line
(the opponent's reply, then our next move, ...). The first move of a text is always `new`.

**stance** (a property of the whole line, so `next` moves copy it from their line's `new` move):
- `chosen`: the line the reflection commits to ("So the move is Qd1#", "The winning idea is …", the stage-4 template's "The queen slides to h8 delivering …")
- `rejected`: the reflection says it fails, is worse, loses, is "tempting but", "also gives check but …"
- `considered`: anything else (mentioned, neutral, hypothetical without a verdict)

### Rules for moves
- SAN/UCI: the span is the notation itself, including `+ # = x` but **not** `! ?`, move numbers (`17.`, `17...`), bold `**`, or brackets.
- Prose moves: the span runs from the moving piece (or the verb, when no piece is named) to the destination / captured piece / promotion piece. Leave out "the" at the start, "to slide" verbs before the piece, and trailing "with check", "delivering mate" (those are claims). Examples:
  - `The winning idea is to slide the {queen to d1}[move new chosen]`
  - `The {queen slides to h8}[move new chosen] <delivering immediate checkmate>[mate]`
  - `{capture the pawn on d7}[move new chosen] <with check>[check]`
  - `the {rook from e1 to f8}[move new chosen]`
  - `the {king simply recaptures}[move next rejected]` (a definite reply without a square still counts)
  - `White can {recapture the queen}[move next rejected]` (the side and "can" stay outside)
  - `{Promoting to a queen with e1=Q}` → just `Promoting to a queen with {e1=Q}[move new rejected]` (when SAN is present, label only the SAN)
- Only label a prose move if it describes **one definite move** that is played (or proposed) in a line. Possibilities over several squares are claims, not moves:
  - `the {king}[piece] <can step to>[flight] {h7}[square] or {g8}[square]` (not a move)
  - `after {Qxg6+}[move new rejected] the {king retreats to e7}[move next rejected]` (a move)
- A move inside a piece-placement phrase is not a move: "the queen on h7" is `{queen}[piece] on {h7}[square]`.

## 2. Claims: `<text>[type]`

Label the **predicate words only**, never the pieces/squares (those are mentions). Include negation words inside the span (`<cannot go to>`, `<is not defended>`, `<no check>`), because the checker reads the negation from the span.

| type | what | examples |
|---|---|---|
| `check` | something gives / is check | `<gives check>`, `<with check>`, `<checks>`, `<is in check>`, `<check>` (noun) |
| `mate` | checkmate / no legal moves | `<checkmate>`, `<delivering immediate checkmate>`, `<mates>`, `<mate>`, `<no legal move>`, `<the game is over>` |
| `wins` | captures / wins material | `<wins>` a pawn, `<wins a knight>`→`<wins>` a `{knight}[piece]`, `<captures>`, `<takes>`, `<hanging>`, `<winning material>` |
| `attacks` | a piece attacks / covers / defends / controls a square or piece | `<covers>`, `<controls>`, `<attacks>`, `<eyes>`, `<hits>`, `<defends>`, `<protects>`, `<guards>`, `<is undefended>`, `<pinned>` |
| `flight` | where a king (or piece) can or cannot go | `<cannot go to>`, `<can escape to>`, `<is boxed in>`, `<is hemmed in>`, `<has no escape>`, `<is trapped>`, `<escape routes>`, `<flight squares>` |

Keep claim spans short (1-4 words). Don't label vague strategy ("strong", "active", "pressure", "fastest", "decisive", "the attack fizzles") — those are not checkable.
A claim word that is itself a prose move (`capture the pawn on d7`) is part of the move, not a claim.
`+` and `#` inside SAN are not separate claims.

## 3. Full example

Original:
```
White has a queen on h7 and a pawn on d7. The tempting Qxg6+ gives check but after the king retreats to e7 the attack fizzles. The winning idea is to capture the pawn on d7 with check: the king cannot go to e7 because the queen on h7 covers that square. So the move is Qxd7#.
```
Labelled:
```
White has a {queen}[piece] on {h7}[square] and a {pawn}[piece] on {d7}[square]. The tempting {Qxg6+}[move new rejected] <gives check>[check] but after the {king retreats to e7}[move next rejected] the attack fizzles. The winning idea is to {capture the pawn on d7}[move new chosen] <with check>[check]: the {king}[piece] <cannot go to>[flight] {e7}[square] because the {queen}[piece] on {h7}[square] <covers>[attacks] that square. So the move is {Qxd7#}[move new chosen].
```

## 4. Decisions from the pilot batch (follow these)
- **Where a prose move starts**: at the moving piece (with its colour if attached: `{Black's rook on d6 can immediately capture the new queen}`) or, when no piece is named, at the verb. A side name + modal is NOT part of it: `White can {recapture the queen}`, `forcing White to {recapture with the b-pawn}`. The from-square may be inside (`{knight on a5 can instead leap to b7}`).
- **King possibilities are claims, other replies are moves**: where a *king* can / cannot / could go (`can step to d7`, `can escape to f2`, `cannot capture it`) is `flight` + squares, even for a single square. A definite king move in a line (`the {king retreats to e7}`, `the {king is forced to f2}`, `{Kxg5}`) is a move. Non-king replies stated as possible (`White can {recapture the queen}`) are moves.
- When SAN is right next to the prose describing the same move, label only the SAN.
- Blocking / occupying / sealing a king's squares (`blocks`, `seals`, `is occupied by`) → `flight`. `is pinned`, `is (un)defended`, `is aimed at`, `cannot reach` (a square) → `attacks`. `has no legal square` → `flight`; `no legal move(s)` → `mate`.
- Mate-quality phrases are mate claims: `<this mate>`, `<the fastest and most direct mate>`, `<ends the game>`.
- `back rank`, `first rank` → rank; `long diagonal` → diagonal; pawns named by file (`b-pawn`, `f5-pawn`) → one piece span.
- Moves made before the puzzle position ("the queen's earlier advance to f7") → piece + square, not a move.
- Arrow notation (`f2→d1`) is a move. Leading `...` stays outside the span.
- Alternatives in a dash/or list ("the king slips away—g4+ or g6"): the first is `next`, the others `alt` (see section 5).
- Unlabelled: vague words (`escapes` with no square, `mating net`, `material is even`, `bearing down on`).

## 5. Frames and alternative replies (second pass, required)

The checker reads every claim and piece placement in a specific position, so each one carries a **frame**:

- `now`: about the current puzzle position, before any move ("White has a `{queen}[piece now]` on c2", "Black's `{king}[piece now]` `<is boxed in>[flight now]`").
- `after`: about the position **after the last move mentioned before it**, in text order. It doesn't matter whether that move was chosen or rejected, or whether it was SAN or prose.

Every `piece` gets `[piece now]` or `[piece after]`. Every claim gets `[check after]`, `[flight now]` and so on. Squares, files, ranks and diagonals get no frame.

How to decide: ask "in which position is this sentence true or false?"
- `The winning idea is to {slide the queen to d1}[move new chosen]. The {king}[piece after] <cannot go to>[flight after] {d2}[square] because the {queen}[piece after] <covers>[attacks after] it.` The second sentence names no move, but it describes the position after Qd1.
- `{Rf8}[move new chosen] <delivering checkmate>[mate after]: the {knight}[piece after] on {g5}[square] <covers>[attacks after] {h7}[square], the {rook}[piece after] on {f8}[square] ...` The rook is only on f8 after the move, so the whole description is `after`.
- An opening description before any move is `now`. A summary after the moves that clearly describes the starting position ("the king on h7 is boxed in by its own pawns", where those pawns are unaffected) is `after` if a move was just discussed. Only use `now` when the text explicitly refers back to the current position ("currently", "right now", "in the position").
- `+ #` inside SAN carry no frame (they're not claim spans).

**Moves: `alt`.** A move gets line `alt` when it is played **from the same position as the move mentioned just before it**: another reply to the same move, not a continuation of it.
- `{Qxc3+}[move new rejected] <gives check>[check after] but White can {recapture the queen}[move next rejected], and the {king simply escapes to d2}[move alt rejected]`: two alternative replies to Qxc3+.
- `the king slips away—{g4+}[move next rejected] or {g6}[move alt rejected]`
- Use `next` only when the move really continues from the position after the previous move. A new candidate from the puzzle position is still `new`.

## 6. Decisions from v3.1 (consistency pass)
- A king capture stated as a possibility ("White can capture the rook with the king", "the king could simply take it") is always `flight` on the modal + verb, with the pieces as mentions, like every other king possibility. Only a definite king capture in a line is a move.
- A piece's colour belongs inside its span when written right before it: `{White's bishop}`, `{black king}`.
- A piece on a line is a piece + the line: `{White's bishop}[piece after] sits on the {a4-e8}[diagonal] diagonal`; the checker reads it as a placement.
- "The g2 pawn" is `{g2}[square] {pawn}[piece ...]`; the checker reads square + piece as a placement.

## 7. Edge cases
- Text cut off mid-sentence: label up to where it stops; never complete it.
- Garbled text ("the f6- pawn", "fxc2 wins a knight"): still label any square / move / piece you can see; `fxc2` is a move even if nonsense.
- Characters `{ } < > [ ]` never appear in a label's text; if the original contains them, leave those characters unlabelled.
- Spans never overlap and never start or end with a space.
