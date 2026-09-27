# chess-moves-from-text

A small model that reads chess commentary and finds every **move** and every **square, file, rank and diagonal** in it, with exact character positions. It's built for the chess-tutor app: the tutor's explanations ("Stockfish prefers Qxd5 over Nb8. The game continued Nb8 Nf3 c6…") become interactive, so clicking a move shows the board after it, and clicking a square or a file highlights it.

The model is [LiquidAI/LFM2.5-Encoder-230M](https://huggingface.co/LiquidAI/LFM2.5-Encoder-230M) fine-tuned for token classification. It only *detects*. Turning detections into links, highlights and move replays is the app's job.

![Demo: hovering dxc6 shows the series Nb8 → Nf3 → c6 → dxc6 on the board](docs/demo.png)

## What it finds

Every span gets one of six types:

| type | example | clicking it in the app… |
|---|---|---|
| `START` | "prefers **Qxd5**" | shows the board after: current position → this move |
| `CONT` | "continued Nb8 **Nf3 c6**" | shows the board after: current position → the series' START → … → this move |
| `SQUARE` | "the pawn on **d5**" | highlights the square |
| `FILE` | "the **c-file** is open" | highlights the column |
| `RANK` | "a rook on the **7th rank**" | highlights the row |
| `DIAGONAL` | "the **a1-h8** diagonal" | highlights the diagonal |

The *current position* is the one before the move being discussed. A single move is a `START` with nothing after it; a series (a "line") is a `START` followed by `CONT`s:

```
Stockfish prefers Black's [[Qxd5]] over [[Nb8]]. The game continued [[Nb8]] [[>Nf3]] [[>c6]] [[>dxc6]].
                          START         START                       START   CONT     CONT    CONT
```

Clicking `dxc6` replays Nb8 → Nf3 → c6 → dxc6 from the current position.

Why a model and not a regex: `e4` is a move in "White played e4", a square in "the pawn on e4", and part of a series in "the game continued Nb8 Nf3 e4"; "the c-file" is a file but "the PGN file" is nothing. Only the context tells them apart.

## Model v1 results

Scored on 184 test texts the model never trained on: 150 from sentence templates held out of training, plus 34 hand-written ones, including real chess-tutor output. A span counts only if its start, end **and** type are exactly right (`python scripts/evaluate.py`).

| | F1 |
|---|---|
| **all spans, correct type** | **0.944** |
| all spans, any type (detection) | 0.997 |
| START / CONT | 0.932 / 0.947 |
| SQUARE | 0.936 |
| FILE / RANK / DIAGONAL | 1.00 / 1.00 / 1.00 |

It almost never misses a reference; nearly all mistakes are about the **type**:

- **Unfamiliar phrasings of a series.** In "I would start with e5, and only after **dxe5** play h4", `dxe5` is read as a START instead of a CONT.
- **A square named with a verb right after a series** ("…Nxe5, the knight eyes **f7**", "hitting **d5**") is often read as the next move of that series. "The pawn on d5" works.
- A square at the start of a list ("squares like **d8** and c1") is sometimes read as a move.
- Castling written with zeros (`0-0`) and move numbers right after a result (`0-1; 9. Bxb5`) can confuse it.
- The labels can't express side branches ("instead of Ne2, Nf3 was also possible") or lines from another position (an opening from move 1). The app should replay each series with a chess library and only link it if every move is legal.
- Ranks are often counted from the speaker's side ("Black's rook on the 7th rank" is White's 2nd rank).

The fix for most of these is more varied training data: add the failing phrasings to `data/generate_dataset.py` or `data/handwritten.txt` and retrain.

## Using the model

### 1. Download it

The weights (~920 MB) live in the [GitHub release](https://github.com/manoletre/chess-moves-from-text/releases/tag/model-v1.0.0), not in git:

```bash
gh release download model-v1.0.0 --pattern 'chess-move-tagger-v1.zip'
unzip chess-move-tagger-v1.zip        # creates ./chess-move-tagger/
```

### 2. Run the demo web app

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r app/requirements.txt
uvicorn app.server:app --port 8000     # from the repo root
open http://localhost:8000
```

Paste any text and hover the highlights: locations light up on the board, and moves show the series you'd replay by clicking them. The page also documents the model. It runs on CPU and uses about 1.1 GB of memory; the first request after startup can be slow while the model warms up.

### 3. Or call it from Python / HTTP

```python
from chess_tagger import ChessTagger

tagger = ChessTagger("chess-move-tagger")
tagger.tag("The game continued Nb8 Nf3, and the pawn on d5 falls.")
# {"spans": [{"start": 19, "end": 22, "type": "START", "text": "Nb8", "series": 0, "index": 0},
#            {"start": 23, "end": 26, "type": "CONT",  "text": "Nf3", "series": 0, "index": 1},
#            {"start": 44, "end": 46, "type": "SQUARE", "text": "d5"}],
#  "series": [[...Nb8, ...Nf3]], "highlights": [...d5], "truncated": false}
```

The web app exposes the same thing as `POST /api/tag` with `{"text": "..."}`. Texts longer than 512 tokens are cut off (`"truncated": true`).

## How it works

1. **Tokenize.** `Qxd5` becomes ` Q`, `x`, `d`, `5`.
2. **Encode.** The LFM2.5 encoder reads all tokens at once, in both directions, and turns each one into 1024 numbers describing it in context.
3. **Classify.** A linear head (1024 × 13) scores 13 labels per token in the BIO scheme: `O` (nothing) or `B-`/`I-` (first/other token of a span) + one of the six types.
4. **Decode.** Consecutive tokens are merged back into character spans, and stray punctuation (`,` `!?` `**`) is trimmed, keeping `+` `#` `=`.

## Training your own version

1. **Data** (optional, since the files are already committed):
   ```bash
   pip install -r requirements.txt
   python data/generate_dataset.py
   ```
2. **Training** happens in Google Colab on a free T4 GPU. Open `notebooks/train_chess_move_tagger.ipynb` in Colab (*File → Upload notebook*), choose *Runtime → Change runtime type → T4 GPU*, and run all cells. When asked, upload `data/train.jsonl`, `data/val.jsonl` and `data/test.jsonl`.
3. The notebook saves the fine-tuned model to Google Drive (`MyDrive/chess-move-tagger`).

The notebook is written for a first-time encoder trainer. Every step (tokens, BIO labels, the classification head, the loss, the training loop, evaluation) is explained right above the code that does it.

## Layout

```
app/
  server.py             # FastAPI demo: serves the page + POST /api/tag
  static/               # the demo page (HTML/CSS/JS), including the model documentation
chess_tagger/
  model.py              # load the trained model and tag text (mirrors the notebook)
scripts/
  evaluate.py           # span-level precision / recall / F1 on a dataset split
data/
  generate_dataset.py   # synthetic data: real legal moves (python-chess) + chess-tutor-style templates
  handwritten.txt       # hand-labelled examples, test set only; add real chess-tutor output here
  train.jsonl           # 1000 examples
  val.jsonl             #  150 examples, same templates as train (used to pick the best epoch)
  test.jsonl            #  184 examples: held-out templates + handwritten.txt
notebooks/
  train_chess_move_tagger.ipynb
```

## Data format

One JSON object per line. `spans` are character positions: `text[19:22] == "Nb8"`. `marked` is the same thing in human-readable form, with a prefix per type:

| markup | type | mnemonic |
|---|---|---|
| `[[Qxd5]]` | START | |
| `[[>Ne2]]` | CONT | `>` continues |
| `[[@e4]]` | SQUARE | `@` at |
| `[[\|c-file]]` | FILE | `\|` vertical line |
| `[[_7th rank]]` | RANK | `_` horizontal line |
| `[[/a1-h8]]` | DIAGONAL | `/` diagonal line |

```json
{"text": "The game continued Nb8 Nf3 c6, and the pawn on d5 falls.",
 "spans": [[19, 22, "START"], [23, 26, "CONT"], [27, 29, "CONT"], [47, 49, "SQUARE"]],
 "marked": "The game continued [[Nb8]] [[>Nf3]] [[>c6]], and the pawn on [[@d5]] falls.",
 "source": "handwritten"}
```

To add your own examples, write them in this markup in `data/handwritten.txt`, one per line (`\n` for line breaks).

## Design notes

- **Only resolvable locations are labelled.** Each span is the smallest text that identifies the location: `c-file`, `7th rank`, `a1-h8`. Vague references like "the long diagonal" or "the back rank" stay unlabelled. Evaluations (`+1.3`), results (`1-0`) and look-alikes (`A4`, `B12`) are mixed into the training data unlabelled, on purpose.
- **Annotations are not part of the move.** For `Nf3!?` or `**Qxd5**`, the span is `Nf3` / `Qxd5`. Check (`+`), mate (`#`) and promotion (`=Q`) *are* part of the move.
- **Loading the encoder.** `AutoModel.from_pretrained("LiquidAI/LFM2.5-Encoder-230M")`, as the model card suggests, silently loads *random* weights with current `transformers`: the checkpoint's parameter names carry an `lfm2.` prefix that the bare body doesn't expect. The notebook loads `AutoModelForMaskedLM` instead and uses its `.lfm2` body. For inference, `chess_tagger` builds the architecture from the config alone and copies the fine-tuned weights in one tensor at a time, so memory peaks at ~1× the model size.

## License

The fine-tuned weights are a derivative of LFM2.5-Encoder-230M and fall under the [LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2.5-Encoder-230M/blob/main/LICENSE).
