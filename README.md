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

## What it finds (v3)

```
{Qxc3+}[move new rejected] <gives check>[check after] but White can {recapture the queen}[move next rejected],
and the {king simply escapes to d2}[move alt rejected]. The winning idea is to slide the {queen to d1}[move new chosen]:
the {king}[piece after] <cannot go to>[flight after] {d2}[square] because the {queen}[piece after] <covers>[attacks after] it.
```

| head | labels | read on |
|---|---|---|
| **mention** (BIO) | `MOVE`, `PIECE`, `SQUARE`, `FILE`, `RANK`, `DIAGONAL` | every token |
| **line** | `NEW` from the current position · `NEXT` after the previous move · `ALT` another reply from the same position as the previous move | MOVE spans |
| **stance** | `CHOSEN` · `REJECTED` · `CONSIDERED` (a property of the whole line) | MOVE spans |
| **frame** | `NOW` the current position · `AFTER` the position after the last move mentioned before it | PIECE spans and claims |
| **claim** (BIO) | `CHECK`, `MATE`, `WINS`, `ATTACKS` (covers, controls, defends, pinned…), `FLIGHT` (can / cannot go to, boxed in…) | every token |

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

Plus `chosen` (the **last** CHOSEN move from the puzzle position, i.e. the conclusion, as UCI) and `matches_played`. `resolve(board, text)` keeps its `(move, status)` shape; `resolve_move(board, text, answered)` also returns the wrong-square note.

`web/verify.js` is a port with the same rules; on all 1,000 gold-labelled texts it reaches the same verdict as `verify.py` for every one of 17,412 facts and every chosen move.

**Using it as a reward.** Suggested, not built in: score the share of *false* facts per reflection (tactical kinds weighted most), cap how many facts count so listing legal moves can't farm reward, require `matches_played`, and treat a high share of *unverifiable* facts as a smell rather than a safe neutral, or RL will learn to write vague prose the rules can't check. Keep the puzzle outcome as the dominant term.

## Results (model v3.1)

Test texts the model never trained on: **157 real reflections** from 129 puzzles that are not in training, **106 held-out synthetic reflections** (other styles, written for v3 and v3.1), the **294 v2 commentary** test texts, and the **probe**: 400 templated reflections whose wording comes from phrase lists the model never saw in training (`scripts/probe.py`). A span counts only if its start, end **and** type are exactly right; line / stance / frame accuracy is measured on the spans found exactly (`python scripts/evaluate.py --verify`).

| | real reflections | synthetic | commentary (v2 test) |
|---|---|---|---|
| **mentions, F1** | **0.985** | 0.976 | 0.977 |
| MOVE / PIECE / SQUARE | 0.967 / 0.983 / 0.992 | 0.974 / 0.969 / 0.992 | 0.988 / 0.949 / 0.913 |
| **claims, F1** | **0.930** | 0.826 | – |
| CHECK / MATE / WINS | 0.955 / 0.873 / 0.886 | 0.899 / 0.886 / 0.883 | – |
| ATTACKS / FLIGHT | 0.955 / 0.934 | 0.800 / 0.687 | – |
| line / stance / frame accuracy | 0.997 / 0.997 / 0.989 | 0.972 / 0.972 / 0.995 | 0.978 / – / – |

**v3 → v3.1 on wording the model hasn't seen** (recall of span + type + attributes on the probe, and F1 on the held-out synthetic texts):

| | v3 | v3.1 |
|---|---|---|
| probe: moves | 0.770 | **0.963** |
| probe: claims attacks / check / mate / flight / wins | 0.62 / 0.64 / 0.43 / 0.54 / 0.91 | **0.87 / 0.77 / 0.72 / 0.92 / 1.00** |
| synthetic: mentions / claims F1 | 0.911 / 0.729 | **0.976 / 0.826** |
| real reflections: mentions / claims F1 | 0.978 / 0.936 | 0.985 / 0.930 |

**What the reward sees.** On the 263 test texts with a FEN, `verify.py` reaches the same verdict from v3.1's labels as from the hand labels for **94.5% of 4,221 facts** (v3: 90.7%), and the same chosen move in **98.5%** of texts (v3: 98.9%). The unverifiable share is 0.080 on predicted labels vs 0.070 on gold ones.

The 4-bit browser model scores 0.985 / 0.922 (mentions / claims) on real reflections and 0.977 on commentary.

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
                           # templated.jsonl (v3.1): 500 from scripts/probe.py, train half of the phrase lists (never in test)
  probe/probe.jsonl        # 400 templated reflections from the probe half of the phrase lists: tests unseen wording
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

The weights live in the [GitHub release](https://github.com/manoletre/chess-moves-from-text/releases/tag/model-v3.1.0), not in git:

```bash
gh release download model-v3.1.0 --pattern 'chess-move-tagger-v3.1.zip'
unzip chess-move-tagger-v3.1.zip        # creates ./chess-move-tagger/ (~920 MB, for Python)
```

`chess-move-tagger-v3.1-web.zip` is the 4-bit browser model; unzip it into `web/` to get `web/model/`.

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
- **Unseen wording is still the weak spot.** On the probe, mate and check claims phrased in new ways are found 72% and 77% of the time, and synthetic flight claims score 0.69 F1. A reward that only penalises *found* false claims can be gamed by phrasing claims the tagger misses; watch for it in RL samples, and add the new phrasings to the training data.
- **The probe is templated**: it tests wording, not sentence structure, so its numbers are an upper bound for truly free text.
- **Linking is rule-based.** `verify.py` attaches a claim to the nearest move/piece before it in its clause (plus passive voice and "it / that square"); this is not measured against gold links. Long sentences with several subjects can get linked wrong.
- **Claims after an unplayable move are checked in the last real position.** That turns most of them from unverifiable into false, which is usually right (the claimed position doesn't exist), but it means a hallucinated move costs more than the single false `legal_move` fact.
- **One source model.** The real reflections all come from one LFM2.5-1.2B chessLLM lineage (SFT and GRPO).
- **Not labelled:** evaluations ("best", "decisive", "wins the initiative"), threats that aren't checks, and lines from another position.
- **Commentary pieces** were added to the v2 data with a regex, so `PIECE` on commentary is only as good as that regex (0.95 F1).

## History

- **v3.1** (this version): a probe with unseen wording showed v3 missed many moves and claims phrased differently from the RL model's own reflections; +750 training reflections aimed at those gaps, a consistency pass over the labels (king captures as possibilities, colours inside piece spans), and many checker fixes so that fewer facts end up unverifiable (see "Checking a reflection"). Long texts are tagged in sentence chunks instead of being cut off at 512 tokens.
- **v3**: new label design for LLM reflections: prose moves, stance, alternative replies, pieces, frames and claims, trained on 800 real reflections + synthetic ones + the v2 data. `verify.py`.
- **v2**: series that continue after prose (500 extra commentary examples). Test F1 0.958 on commentary.
- **v1**: moves (START/CONT) and squares, files, ranks, diagonals in chess-tutor commentary.

## License

The fine-tuned weights are a derivative of LFM2.5-Encoder-230M and fall under the [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-Encoder-230M/blob/main/LICENSE).
