# chess-moves-from-text

A small model that reads chess text and pulls out what it says, with exact character positions:

- every **move** (in SAN, UCI, long algebraic *or prose*: "the queen slides to h8"), whether it starts a line, continues one or is an alternative reply, and whether the text **chose** it, **rejected** it or just mentioned it;
- every **piece**, **square, file, rank and diagonal**;
- every checkable **claim** about the board: check, mate, wins material, attacks/covers, where a king can or can't go, and in which position it is meant (**now** or **after** the move just discussed).

It is built for two jobs:

1. **Rewarding chess reflections.** A puzzle-solving LLM ([chessLLM](https://github.com/manoletre)) writes a short reflection before every move. `chess_tagger.verify` replays what the tagger found against the position with python-chess and returns true / false / unverifiable for every fact, so an RL run can reward reflections that contain proper chess, not just the right final move.
2. **Interactive commentary** (the chess-tutor app): clicking a move replays its line on a board, clicking a square, file or diagonal highlights it.

The model is [LiquidAI/LFM2.5-Encoder-230M](https://huggingface.co/LiquidAI/LFM2.5-Encoder-230M) fine-tuned for token classification, with one output layer split into five heads. It only *detects*; checking is deterministic code.

**Live demo: [chess-annotator.vercel.app](https://chess-annotator.vercel.app)**. The model runs in your browser; nothing is sent to a server.

## What it finds (v3.3)

```
{Qxc3+}[move new rejected] <gives check>[check after] but White can {recapture the queen}[move next rejected],
and the {king simply escapes to d2}[move alt rejected]. The winning idea is to slide the {queen to d1}[move new chosen]:
the {king}[piece after] <cannot go to>[flight after] {d2}[square] because the {queen}[piece after] <covers>[attacks after] it.
```

| head | labels | read on |
|---|---|---|
| **mention** (BIO) | `MOVE`, `PIECE`, `SQUARE`, `FILE`, `RANK`, `DIAGONAL`, `REGION` (center, kingside, queenside, corner; v3.3) | every token |
| **line** | `NEW` from the current position · `NEXT` after the previous move · `ALT` another reply from the same position as the previous move | MOVE spans |
| **stance** | `CHOSEN` · `REJECTED` · `CONSIDERED` (a property of the whole line) | MOVE spans |
| **frame** | `NOW` the current position · `AFTER` the position after the last move mentioned before it | PIECE spans and claims |
| **claim** (BIO) | `CHECK`, `MATE` (incl. "it's over", "the king is dead", "Black resigns"), `WINS`, `ATTACKS` (covers, controls, defends, pinned…), `FLIGHT` (can / cannot go to, boxed in, "the king simply steps away"…), `EVAL` ("decisive", "the winning idea", "Black is lost", "a mistake"; v3.2; "the attack fizzles", "lets White consolidate"; v3.3) | every token |

Claims are the predicate words only, negation included (`<cannot go to>`); their subject and objects are the mentions around them. The full labelling rules, with the edge cases decided along the way, are in [`data/reflections/LABELING.md`](data/reflections/LABELING.md).

Why a model and not a regex: `e4` is a move in "White played e4" and a square in "the pawn on e4"; "the queen slides to h8" is a move but "the queen on h8" is a piece on a square; "Qxc3+ gives check but…" is a rejected candidate and "So the move is Qd1#" the chosen one. Only the context tells them apart.

## Checking a reflection

```python
from chess_tagger import ChessTagger
from chess_tagger.verify import verify

tagger = ChessTagger("chess-move-tagger")
fen = "4k3/p1Q2p2/4p3/4P3/3P3p/P1P1KP2/1PqpR3/8 b - - 1 40"
text = "Qxc3+ gives check but White can recapture the queen. The winning idea is to slide the queen to d1. So the move is Qd1#."
report = verify(fen, text, tagger.predict([text])[0], played="c2d1")
for f in report["facts"]:
    print(f["kind"], repr(f["text"]), f["verdict"], f["detail"])
# legal_move 'Qxc3+' True c2c3
# check 'Qxc3+' True + in notation
# legal_move 'recapture the queen' None ambiguous   <- bxc3 or Qxc3: can't tell which
# legal_move 'queen to d1' True c2d1
# legal_move 'Qd1#' True c2d1
# mate 'Qd1#' False # in notation                  <- the reflection claims a mate that isn't there
# check 'gives check' True
report["chosen"], report["matches_played"]   # 'c2d1', True
```

What `verify` checks (all with python-chess, in the claim's frame):

| fact | true when |
|---|---|
| `legal_move` | the move (any notation; prose resolved against the legal moves) is legal where its line says it is played. "Recaptures" / "takes it" is resolved on the square of the move it answers; "checks" / "delivers checkmate" inside a prose move narrows it down. Still ambiguous → unverifiable |
| `check` / `mate` | the frame position is check / checkmate (also `+` / `#` in SAN, and "delivers checkmate" inside a prose move) |
| `capture` | an `x` move really captures |
| `wins` | the move it's about captures (the named piece, if one is named) |
| `attacks` | the subject piece attacks every target (or none, if negated). Targets: squares, pieces, a file/rank/diagonal ("controls the e-file": a line piece on it), or the square named just before ("covers **it**", "that square", "both"). Passive voice ("g1 is covered by the rook on g2") swaps subject and object. "Undefended" = no defenders, "pinned" = pinned to its king, "cannot reach" excludes squares its own pieces stand on. A named piece that isn't there ("the rook on c7 covers…") makes the claim false |
| `flight` | the king can reach the named squares (or can't, for "cannot go to", "blocks", "boxed in"…); with no squares, "has no escape" = no king moves. A king with no colour named is the defender's |
| `placement` | "the queen on h7", "the g2 pawn", "the bishop on the a4-e8 diagonal": that piece (and colour, if named) is there in the frame. A move written from the wrong square ("the bishop on e5 slides to f7" with the bishop on d5) resolves, and adds a false placement |

Decoding guard: a MOVE span with nothing move-like in it ("rook on" in "the rook on g2 delivers mate") is turned into a PIECE (a bare piece word) or dropped (a single other word), so a broken span can't become the chosen move.

How positions are chosen: a claim is checked in its frame (now, or after the last move before it). After a move that **can't be played** (illegal or hallucinated), it is checked in the last real position instead, and its detail says "(checked before the unplayable move)". Before v3.1 those claims were all unverifiable; on gold labels this cut the unverifiable share from 0.23 to 0.07. A piece called by the square it just left ("Rxh7 … the rook on g7 covers g8") is the piece that moved.

| `eval` | (v3.2) needs `best=`, the puzzle's solution move. A positive evaluation ("decisive", "the winning idea", "unstoppable") is true if the move it is about (nearest move in its sentence, else the chosen move) is the solution; a negative one ("a mistake", "too slow", "falls short") if it isn't; "Black is lost" / "White is winning" are about sides. Without `best`, unverifiable |

| `attacks` / `flight` / `placement` on a **region** | (v3.3) a region is a square set: center = d4, e4, d5, e5 (c3–f6 where a king escapes to it, or "extended center"); kingside = files e–h; queenside = files a–d; corner = the four 2×2 corner blocks. "Puts pressure on the center": the piece attacks (or stands on) one of its squares; "the king escapes to the kingside": it can reach one; "stuck in the corner": it is there and every king move stays there |

Plus `chosen` (the **last** CHOSEN move from the puzzle position, i.e. the conclusion, as UCI) and `matches_played`. `resolve(board, text)` keeps its `(move, status)` shape; `resolve_move(board, text, answered)` also returns the wrong-square note.

**Refutations and self-contradiction (v3.3).** After RL the puzzle model often dismisses a candidate with a stock refutation ("Rb3+ gives check but the king simply steps away") and then plays that very move. `verify` now returns:

- `report["refutations"]`: the facts that dismiss a rejected candidate: the opponent's replies in its line ("Black can recapture the queen", "White can block with the rook"), the king getting away after it ("simply steps away", "escapes to the center"), a negated mate/check after it ("no mate follows") and a negative evaluation of it ("the attack fizzles", "leaving White without a decisive follow-up"). They are ordinary entries of `report["facts"]` (kind `legal_move`, `flight`, `check`, `mate`, `eval`) with an extra `"refutes"`: the candidate's UCI. `report["false_refutations"]` counts the false ones.
- `report["contradiction"]`: `{"move", "rejected_at", "chosen_at", "rejected_text", "chosen_text", "refuted_by"}` when a candidate is rejected and later chosen as the same move (same UCI) with no "actually", "on second thought", "wait"… in between; else `None`.

An opponent's reply stated as possible that fits several legal moves ("recapture the queen" with two recaptures) is true, and "block with the rook (on e2)" resolves to an interposing move.

`web/verify.js` is a port with the same rules; `scripts/check_verify_js.py` runs both on 2,910 gold-labelled texts (labels, synthetic, refute probe): the same verdict for every one of 34,621 facts, and the same chosen move, contradiction and false-refutation count for every text.

**Auditing for reward hacking (v3.2).** A process reward only penalises claims the tagger finds, so a policy can learn to phrase false confidence in wording the tagger misses. `chess_tagger.audit.audit(text, tagged, report)` checks every reflection against a fixed lexicon of mate / evaluation / check wording that is independent of the model, and returns the hits no tagged claim covers, plus flags (`untagged_mate`, `untagged_eval`, `no_checkable_facts`, …). Log the flag rates during RL: a rising rate means the policy found phrasings the tagger misses. Collect those samples and add them to the training data. The demo shows the same audit (`web/audit.js`, generated from the Python lexicon by `scripts/sync_audit_js.py`).

**Using it as a reward.** Suggested, not built in: score the share of *false* facts per reflection (tactical kinds weighted most), cap how many facts count so listing legal moves can't farm reward, require `matches_played`, and treat a high share of *unverifiable* facts as a smell rather than a safe neutral, or RL will learn to write vague prose the rules can't check. Keep the puzzle outcome as the dominant term.

## Results (model v3.3)

Test texts the model never trained on: **194 real reflections** from puzzles not in training (157 from cur-v2, 37 from cur-v3 stage 0/1 added in v3.3), **153 held-out synthetic reflections** (other styles, including faked-confidence ones written for v3.2), the **294 v2 commentary** test texts, and two **probes** of templated reflections whose wording comes from phrase lists the model never saw in training (`scripts/probe.py`): 500 general ones and 300 refutation/region ones (v3.3). A span counts only if its start, end **and** type are exactly right (`python scripts/evaluate.py --verify`).

| | real reflections | synthetic | commentary (v2 test) |
|---|---|---|---|
| **mentions, F1** | **0.987** | 0.980 | 0.987 |
| **claims, F1** | **0.939** | 0.869 | – |
| CHECK / MATE / WINS | 0.969 / 0.921 / 0.883 | 0.916 / 0.864 / 0.894 | – |
| ATTACKS / FLIGHT / EVAL | 0.963 / 0.942 / 0.890 | 0.863 / 0.758 / 0.923 | – |
| line / stance / frame accuracy | 0.995 / 0.996 / 0.989 | 0.984 / 0.979 / 0.997 | 0.975 / – / – |

(The v3.3 test labels include the new refutation, region and pressure spans; v3.2 scores 0.974 / 0.853 on the same real reflections.)

**Faked confidence.** 40 ways of claiming mate ("finishes it", "the king is dead", "Black is lost", "is #", "is decisive", …), each written after a move that is *not* mate and *not* the solution: v3.1 left 20 of them untagged (free for a policy); **v3.2 tags and checks 39 of 40 as false**. Written after the real mate, 38 of 40 come out true. The one it misses ("the attack crashes through") is too vague to check.

**Probe (unseen wording):** moves 0.97, claims attacks 0.95 / flight 0.86 / EVAL 0.85 / mate 0.73 / check 0.82 / wins 1.00 (v3.2: 0.96; 0.93 / 0.89 / 0.82 / 0.75 / 0.68 / 1.00).

**Refutation probe (v3.3, unseen wording):** regions 1.00, king-escape flights 0.96, moves 0.93, dismissals (EVAL) 0.72, pressure (ATTACKS) 0.69. Checked with `verify.py` against the construction labels: refutation facts get the same verdict for **81%** (v3.2: 6.5%), the contradiction flag agrees for **98.3%** (v3.2: 86%), and **109 of the 118** texts with a false refutation get one (v3.2: 4).

**On the chessLLM cur-v3 stage-1 benchmark** (1,221 reflections that play a legal move; not in training): 39.1% reject a candidate and then play it (stage 0: 13.5%), and 49.5% contain a false refutation (v3.2 labels: 39.5%). Of the 150 such contradictions whose played move was the correct solution, **134 now carry a false refutation of it** (v3.2: 77), mostly a dismissal checked against the solution.

**What the reward sees.** On the 347 test texts with a FEN, `verify.py` reaches the same verdict from v3.3's labels as from the hand labels for **95.5% of 5,732 facts**, and the same chosen move in **98.6%** of texts (v3.2 on the same texts and rules: 89.6% / 99.1%; on its own v3.2 test set: 94.3% / 99.0%). The chosen-move misses are 5 synthetic garbled / coach texts (v3.2: 3); on real reflections both get every chosen move. (EVAL facts are unverifiable here because the test set has no solution moves; the RL loop passes them.)

The 4-bit browser model scores 0.982 / 0.914 (mentions / claims) on real reflections and 0.991 on commentary.

v3.2 → v3.3 (on the v3.3 test set): real reflections 0.974 / 0.853 → 0.987 / 0.939, commentary 0.993 → 0.987.

v3.1 → v3.2: real reflections 0.985 / 0.930 → 0.984 / 0.925 (the claims now include EVAL), commentary 0.977 → 0.993, same chosen move 0.985 → 0.990.

## Data

```
data/
  train.jsonl, val.jsonl, test.jsonl   # v3, built by scripts/build_dataset.py from the three sources below
  reflections/
    raw.jsonl              # 800 real reflections from the chessLLM puzzle model, with FEN and the move played
    labels/b00..b15.jsonl  # their labels in the v3 markup (hand-labelled, see LABELING.md)
    synthetic/             # synthetic.jsonl: 200 reflections in other styles (UCI, prose-only, negation, coach, garbled, wrong claims)
                           # paraphrase_a/b.jsonl (v3.1): 250 more, aimed at what the probe found missing: claim wording,
                           #   alternative replies, passive voice and pronouns, pieces on lines, RL-drift styles, mixed notation
                           # templated.jsonl: 600 from scripts/probe.py, train half of the phrase lists (never in test)
                           # fake_claims_a/b.jsonl (v3.2): 200 reflections that fake confidence (false mates, "decisive",
                           #   "Black is lost", repetition, vague-but-confident) next to honest ones
                           # (v3.2 also labelled game-ending wording and evaluations in all earlier labels: scripts/relabel_v32.py)
  probe/probe.jsonl        # 500 templated reflections from the probe half of the phrase lists: tests unseen wording
    LABELING.md            # the labelling guide
    batches/               # what each labeller got (id, FEN, side to move, played move, text)
  commentary/              # the v1/v2 chess-tutor data (generate_dataset.py, handwritten.txt, {train,val,test}.jsonl)
```

- **Real reflections** come from the chessLLM curriculum GRPO run (`cur-v2`): the SFT model before RL (stage 0, which weighs several candidates) and stages 1–4 (after RL, mostly one templated mate claim), plus samples at temperatures 0.25–1.4 (including garbled ones). They are split **by puzzle**, so no test puzzle is seen in training: 581 train / 62 val / 157 test.
- **Synthetic reflections** cover what the real model rarely writes; a quarter of each category is held out for test.
- **Commentary** (v1/v2 data) keeps the old use case working. It is converted to v3 (START → `MOVE new`, CONT → `MOVE next`, pieces added with a regex) and has no stance, frame or claim labels: those heads are masked out on it instead of being taught "nothing here".

One line of `data/*.jsonl`:

```json
{"text": "...", "mentions": [[start, end, "MOVE", "NEW", "REJECTED", null], [start, end, "PIECE", null, null, "AFTER"], ...],
 "claims": [[start, end, "CHECK", "AFTER"], ...],
 "labelled": "full", "marked": "{Qxc3+}[move new rejected] <gives check>[check after] ...",
 "group": "reflection", "source": "cur-v2-stage0", "fen": "...", "played": "c2d1"}
```

`chess_tagger.markup.parse(marked)` / `render(text, mentions, claims)` convert between the markup and spans.

## Using the model

### 1. Download it

The weights live in the [GitHub release](https://github.com/manoletre/chess-moves-from-text/releases/tag/model-v3.3.0), not in git:

```bash
gh release download model-v3.3.0 --pattern 'chess-move-tagger-v3.3.zip'
unzip chess-move-tagger-v3.3.zip        # creates ./chess-move-tagger/ (~920 MB, for Python)
```

`chess-move-tagger-v3.3-web.zip` is the 4-bit browser model; unzip it into `web/` to get `web/model/`.

### 2. Call it from Python

```python
from chess_tagger import ChessTagger

tagger = ChessTagger("chess-move-tagger")   # CUDA, Apple MPS or CPU
tagger.tag("The queen slides to h8 delivering mate, as the king on h7 cannot go to g8.")
# {"mentions": [{"type": "MOVE", "text": "queen slides to h8", "line": "NEW", "stance": "CHOSEN", "series": 0, ...},
#               {"type": "PIECE", "text": "king", "frame": "AFTER"}, {"type": "SQUARE", "text": "h7"}, ...],
#  "claims": [{"type": "MATE", "text": "delivering mate", "frame": "AFTER"}, {"type": "FLIGHT", "text": "cannot go to", ...}],
#  "series": [...], "highlights": [...], "marked": "...", "truncated": false}
tagger.predict(texts)   # batched, raw spans: [{"mentions": [[s, e, type, line, stance, frame]], "claims": [[s, e, type, frame]]}]
```

`pip install -r requirements-model.txt` first. Texts longer than 512 tokens are cut off.

### 3. The demo page

`web/` is static: the model runs in the browser with [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/), and `web/verify.js` (a port of `verify.py`, with the same rules) checks what it found with [chess.js](https://github.com/jhlywa/chess.js). Paste a FEN and a text, or pick an example. The page keeps two layers apart:

- **What the tagger read** (always shown): the moves, as chips (filled = chosen, struck through = rejected), and every fact the checker reads, as one underline: a claim ("cannot go to") or a piece on a square ("rook on h2"). Pieces and squares a fact refers to light up when you hover it; they aren't painted on their own.
- **What chess.js says** (the "+ chess.js check" toggle): a ✓ / ✗ / ? on every move and fact, verdict-coloured underlines, and a third column in the panel under the text, which lists the chosen move, the lines and every fact with the position it is checked in.

Hover a move to replay its line, or a fact to see the board it's checked on.

```bash
python scripts/export_onnx.py          # chess-move-tagger/ -> web/model/ (4-bit, in 50 MB parts)
python -m http.server 8000 --directory web
cd web && npx vercel deploy --prod     # deploy (the model folder isn't in git, so a git push doesn't deploy it)
```

## Training

```bash
pip install -r requirements-model.txt
python scripts/collect_reflections.py ~/Dev/chessLLM/results   # only to resample raw.jsonl
python scripts/label_tool.py check-all --frames                 # validate the labels
python scripts/build_dataset.py                                 # -> data/{train,val,test}.jsonl
python scripts/train.py                                         # -> chess-move-tagger/ (≈25 min on an M3 Max)
python scripts/evaluate.py --verify                             # test scores + verify.py agreement
```

`scripts/train.py` is the v3 recipe: the encoder plus one linear layer (1024 × 32) whose outputs are split into the five heads, each with its own cross-entropy, ignored (-100) where a text has no label for it. Fully labelled texts are shown twice per epoch, and the epoch with the best mean of mention F1, claim F1 and attribute accuracies on the real validation reflections is kept. `notebooks/train_chess_move_tagger.ipynb` is the step-by-step explanation of token classification from v1/v2 (one head, six types); the ideas carry over.

## Layout

```
chess_tagger/
  model.py      # load the model, tag text (heads, decoding)
  markup.py     # the v3 markup <-> spans
  verify.py     # check a tagged text against a FEN with python-chess
  metrics.py    # span F1 + attribute accuracy
scripts/        # collect_reflections, label_tool, build_dataset, train, evaluate, export_onnx
web/            # the demo page (tagger.js mirrors chess_tagger/model.py)
data/           # see above
```

## Known limitations

- **Labels come from Claude subagents, not humans.** 16 labellers each did 50 real reflections with the guide, more agents wrote the synthetic sets, and a consistency pass fixed the conventions they disagreed on most (see `LABELING.md` section 6). Every text passes the markup checks and they were spot-checked, but no human has reviewed all 1,750 fully labelled texts.
- **Frames follow text order.** Labellers converged on "`now` before the first move, `after` from then on" (plus "already…", "right now…"), so the frame head mostly learned that rule. A claim that sums up a candidate after its replies ("…the king slips away to g4 or g6, so there is no forced mate") is framed after the last reply, not after the candidate.
- **`alt` and `considered` are still rare** (57 and 78 of 6,051 moves).
- **Unseen wording is still the weak spot.** On the probe, mate and check claims phrased in new ways are found 75% and 68% of the time, and synthetic flight claims score 0.71 F1. A reward that only penalises *found* false claims can be gamed by phrasing claims the tagger misses: run `chess_tagger.audit` in the RL loop, watch its flag rates, and add new phrasings to the training data.
- **EVAL needs the solution move.** Without `best=`, every evaluation is unverifiable; with it, "decisive" means "this is the puzzle's solution", which is right for puzzles but not for positions with several good moves.
- **The probe is templated**: it tests wording, not sentence structure, so its numbers are an upper bound for truly free text.
- **Linking is rule-based.** `verify.py` attaches a claim to the nearest move/piece before it in its clause (plus passive voice and "it / that square"); this is not measured against gold links. Long sentences with several subjects can get linked wrong.
- **Claims after an unplayable move are checked in the last real position.** That turns most of them from unverifiable into false, which is usually right (the claimed position doesn't exist), but it means a hallucinated move costs more than the single false `legal_move` fact.
- **One source model.** The real reflections all come from one LFM2.5-1.2B chessLLM lineage (SFT and GRPO).
- **Not labelled:** vague strategy ("wins the initiative", "active", "keeps the pressure" with no target), threats that aren't checks, and lines from another position.
- **Commentary pieces** were added to the v2 data with a regex, so `PIECE` on commentary is only as good as that regex (0.95 F1).

## History

- **v3.3** (this version): refutations. The king getting away without a square ("simply steps away", "escapes") is a `flight` claim, how a candidate is dismissed ("the attack fizzles", "lets White consolidate", "without a decisive follow-up") an `eval` claim, pressure wording with a target ("bearing down on f2") an `attacks` claim, and a new `REGION` mention (center, kingside, queenside, corner) is checked as a square set. `verify` reports refutations, false refutations and a rejected-then-chosen contradiction. 160 new real reflections from cur-v3 stages 0–1, 400 templated refutations, a pass over all earlier labels (`scripts/relabel_v33.py`, LABELING.md section 9). Models trained with v3.2's labels still load (they just never predict REGION).
- **v3.2**: built against reward hacking. Game-ending wording ("finishes it", "the king is dead", "is #") is a `mate` claim, and a new `EVAL` claim covers "decisive", "the winning idea", "Black is lost", "a mistake", checked against the puzzle's solution move. 450 new training reflections, 200 of them written to fake confidence, plus a pass that labelled these in the existing data. `chess_tagger.audit` flags confident wording the tagger misses. Checker: contractions negate ("isn't decisive"), "unstoppable" no longer does, "there is no defence" is a mate claim, not a negated one.
- **v3.1**: a probe with unseen wording showed v3 missed many moves and claims phrased differently from the RL model's own reflections; +750 training reflections aimed at those gaps, a consistency pass over the labels (king captures as possibilities, colours inside piece spans), and many checker fixes so that fewer facts end up unverifiable (see "Checking a reflection"). Long texts are tagged in sentence chunks instead of being cut off at 512 tokens.
- **v3**: new label design for LLM reflections: prose moves, stance, alternative replies, pieces, frames and claims, trained on 800 real reflections + synthetic ones + the v2 data. `verify.py`.
- **v2**: series that continue after prose (500 extra commentary examples). Test F1 0.958 on commentary.
- **v1**: moves (START/CONT) and squares, files, ranks, diagonals in chess-tutor commentary.

## License

The fine-tuned weights are a derivative of LFM2.5-Encoder-230M and fall under the [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-Encoder-230M/blob/main/LICENSE).
