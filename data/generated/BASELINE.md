# v3.3 on generated reflections (baseline, 2026-11-03)

Generator: the snapshot in this folder. 500 texts with the generator's natural wording, 300 with held-out wording forced
wherever a pool has it; games not used in training. Labels from `scripts/generated_labels.py`, scored by
`scripts/eval_generated.py`. The text is treated as correct, so a False verdict is a false alarm.

| | natural wording | held-out wording |
|---|---|---|
| mentions F1 (precision / recall on natural) | 0.571 (0.401 / 0.991) | 0.522 |
| MOVE F1 | 0.742 | 0.793 |
| PIECE F1 | 0.194 | 0.134 |
| SQUARE F1 | 0.187 | 0.108 |
| claims F1 | 0.329 | 0.327 |
| CHECK / ATTACKS | 0.792 / 0.709 | 0.892 / 0.695 |
| MATE / EVAL | 0.015 / 0.000 | 0.000 / 0.000 |
| stance accuracy | 0.320 | 0.321 |
| line / frame accuracy | 0.979 / 0.993 | 0.956 / 0.994 |
| moves replayed to the right UCI | 0.457 | 0.434 |
| chosen move right | 0.978 | 0.967 |
| facts false / unverifiable | 14.2% / 50.2% | 9.6% / 52.1% |
| texts with a false fact | 99.6% | 93.7% |
| rejected, then chosen (phantom contradiction) | 5.0% | 6.3% |

The same 500 texts with the gold labels through `verify.py` give 1.9% false and 17.4% unverifiable facts, 100% right chosen move.

Most common failures on natural wording:
- Prose restating a SAN is read as a move and a piece and a square: 5,513 spurious PIECE, 3,687 spurious SQUARE, about 4,500
  spurious MOVE ("knight heads to e5" next to its "(Ne5)"). This breaks the line replay and produces most false facts.
- 6,170 of 9,573 REJECTED candidate moves are read as CHOSEN: it has never seen a text weigh several candidates at this length.
- Verdicts: only 22% of them overlap any EVAL span. "It's about equal" and "the game stays balanced" are never tagged, and tagged
  ones cover a word ("winning") rather than the phrase.
- Material counts are tagged as WINS ("gained", "material"), captures in prose too ("grabs", "gobbles").
- "out of an attack" and "their pawn on c6 can take it" are missed.

False verdicts on the gold labels are gaps in `verify.py`, not generator bugs (`reflect_metrics.py` also finds 0 truth failures):
- 222: "with check" / "giving check" written before its SAN is read in the position before that move.
- 56: "their pawn on c6 can take it" reads c6 as the destination.
- About 40: "I mate in 5" is judged by whether the position is already checkmate. 2: "undefended" before a piece.

v3.3 on the existing sets (these reproduce the README): real reflections 0.987 / 0.939, synthetic 0.980 / 0.869, commentary 0.987
(mentions / claims F1); verify agreement 95.5%, same chosen move 98.6%; probe moves 0.972, refutation probe moves 0.933.
