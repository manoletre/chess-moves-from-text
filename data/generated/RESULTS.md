# v3.4 on generated reflections (2026-11-04)

Trained from the v3.3 weights: the 3,701 existing training texts (hand-labelled ones twice) plus 1,500 generated ones, 4 epochs
at lr 2e-5, window 1024, on the Mac (about 7 min per epoch; epoch 3 kept). Not released, and the browser model is not rebuilt.

```bash
python scripts/train.py --init chess-move-tagger --out chess-move-tagger-v3.4 --epochs 4 --lr 2e-5 --max-length 1024 \
    --extra-train gen_train.jsonl --extra-val gen_val.jsonl
```

Both columns run with the same `verify.py` and decoder, so v3.3's numbers differ a little from `BASELINE.md`.

Held-out wording forces 58 phrases that never appeared in training wherever a pool has one, on games not used in training, so it
is harder than real output. Natural wording is the generator's own mix, which also contains those phrases.

| | held-out (300) v3.3 | held-out v3.4 | natural (500) v3.3 | natural v3.4 |
|---|---|---|---|---|
| mentions F1 | 0.522 | 0.996 | 0.571 | 0.998 |
| MOVE F1 | 0.793 | 1.000 | 0.742 | 1.000 |
| PIECE F1 | 0.134 | 0.973 | 0.194 | 0.990 |
| SQUARE F1 | 0.108 | 0.945 | 0.187 | 0.979 |
| claims F1 | 0.327 | 0.533 | 0.329 | 0.876 |
| CHECK | 0.892 | 0.850 | 0.792 | 0.945 |
| ATTACKS | 0.695 | 0.787 | 0.709 | 0.919 |
| EVAL | 0.000 | 0.367 | 0.000 | 0.852 |
| MATE | 0.000 | 0.143 | 0.015 | 0.672 |
| FLIGHT (8 and 14 spans) | 0.000 | 0.483 | 0.000 | 0.619 |
| stance | 0.321 | 0.965 | 0.320 | 0.995 |
| line | 0.956 | 0.998 | 0.979 | 0.999 |
| moves replayed to the right UCI | 0.431 | 0.991 | 0.455 | 0.996 |
| chosen move right | 0.967 | 1.000 | 0.978 | 1.000 |
| facts false | 9.0% | 0.4% | 13.8% | 0.1% |
| facts unverifiable (gold labels: about 17.5%) | 52.3% | 18.3% | 50.5% | 18.4% |
| texts with a false fact | 93.7% | 8.0% | 99.6% | 2.8% |
| rejected, then chosen (phantom contradiction) | 6.3% | 5.3% | 5.0% | 1.6% |

Existing sets:

| | v3.3 | v3.4 |
|---|---|---|
| real reflections (194): mentions / claims | 0.987 / 0.939 | 0.987 / 0.938 |
| real reflections: line / stance / frame | 0.995 / 0.996 / 0.989 | 0.996 / 0.996 / 0.992 |
| synthetic (153): mentions / claims | 0.980 / 0.869 | 0.976 / 0.869 |
| commentary (294): mentions | 0.987 | 0.983 |
| verify agreement with gold labels (5,732 facts) | 95.5% | 95.3% |
| same chosen move | 98.6% | 98.8% |
| probe: MOVE / ATTACKS / CHECK | 0.972 / 0.952 / 0.816 | 0.960 / 0.984 / 0.895 |
| probe: EVAL / FLIGHT / MATE | 0.849 / 0.863 / 0.727 | 0.874 / 0.835 / 0.709 |
| refutation probe: MOVE / CHECK / FLIGHT | 0.933 / 0.772 / 0.960 | 0.917 / 0.886 / 0.947 |
| refutation probe: EVAL / ATTACKS | 0.723 / 0.688 | 0.723 / 0.688 |

## Remaining failures

- **Unseen verdict wording is the main gap.** EVAL is 0.367 on held-out wording. Most of the loss is the phrase being cut into
  pieces ("I'd be a bit on" + "foot") or missed ("Neither side has much": 262). This costs little in `verify.py` today, because
  EVAL needs a solution move and the checker reads the verdict from whichever fragment it gets.
- **Unseen "chosen" wording makes phantom contradictions.** On held-out wording, 179 chosen moves are read as rejected, so 5.3% of
  texts look as if they reject the move they then choose. The final chosen move is still right in all of them.
- **Unseen mate wording.** MATE is 0.143 on held-out and 0.672 on natural; the misses are mostly the held-out phrasings ("That's
  mate in N for me", "That runs into mate in N").
- **Unseen attack and check wording.** "goes after" and "escapes an attack" are cut or missed. CHECK on held-out wording is the
  one score below v3.3 (0.892 to 0.850), from "giving check" style phrasing split at "check".
- **Small losses on other text.** Prose moves in unseen wording lose 1.2 to 1.6 points (the probes' MOVE), mentions lose up to 0.4
  points on synthetic and commentary. Real reflections are unchanged.
- **The generated validation scores are not evidence.** Its wording is the training wording, and it reached 1.000 on mentions and
  stance. Only the held-out column tests generalisation, and adding more generated data with the same phrases won't help that.

`verify.py` still has one known gap on correct labels: "the undefended pawn on b4" (2 of 14,177 facts, left as it is).
The generator is a snapshot (`SNAPSHOT.md`): text from a later version is not covered by these numbers.
