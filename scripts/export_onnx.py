"""Export the trained tagger to a 4-bit ONNX model for the in-browser demo (web/).

    pip install onnx onnxruntime onnxscript
    python scripts/export_onnx.py                  # chess-move-tagger/ -> web/model/

The graph takes one text at a time (input_ids of shape [1, n], no padding) and
returns logits of shape [1, n, 32]: all five heads side by side, in the order of
"heads" in tagger_config.json (web/tagger.js splits them). Weights are quantized
block-wise to 4 bits (blocks of 32, embeddings included): ~145 MB instead of 920 MB.
Plain per-tensor int8 was both bigger (231 MB) and worse (v2).

The model is written in 50 MB parts (Vercel's free plan rejects files over 100 MB);
tagger_config.json lists them and the page joins them back together.
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import onnx
import torch
from onnxruntime.quantization.matmul_nbits_quantizer import DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chess_tagger import ChessTagger  # noqa: E402

PART_SIZE = 50_000_000


class SingleText(torch.nn.Module):
    """input_ids -> logits. Without padding the attention mask is all ones, so leave it out."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        hidden = self.model.encoder(input_ids=input_ids).last_hidden_state
        return self.model.classifier(hidden)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=ROOT / "chess-move-tagger", type=Path)
    parser.add_argument("--out", default=ROOT / "web" / "model", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tagger = ChessTagger(args.model, device="cpu")
    example = tagger.tokenizer("The game continued Nb8 Nf3, and the pawn on d5 falls.",
                               return_tensors="pt")["input_ids"]
    with tempfile.TemporaryDirectory() as tmp:
        fp32 = Path(tmp) / "model.onnx"
        with torch.no_grad():
            torch.onnx.export(SingleText(tagger.model).eval(), (example,), fp32,
                              input_names=["input_ids"], output_names=["logits"],
                              dynamic_shapes={"input_ids": {1: torch.export.Dim("tokens", min=2, max=tagger.max_length)}},
                              dynamo=True, external_data=False)
        del tagger

        config = DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True, bits=4,
                                              op_types_to_quantize=("MatMul", "Gather"),
                                              quant_axes=(("MatMul", 0), ("Gather", 1)))
        quantizer = MatMulNBitsQuantizer(onnx.load(fp32), algo_config=config)
        quantizer.process()
        q4 = Path(tmp) / "model_q4.onnx"
        quantizer.model.save_model_to_file(str(q4), use_external_data_format=False)
        data = q4.read_bytes()

    for old in args.out.glob("model_q4.onnx*"):
        old.unlink()
    parts = []
    for i in range(0, len(data), PART_SIZE):
        parts.append(f"model_q4.onnx.{len(parts):02d}")
        (args.out / parts[-1]).write_bytes(data[i:i + PART_SIZE])
    config = json.loads((args.model / "tagger_config.json").read_text())
    config["onnx_parts"], config["onnx_size"] = parts, len(data)
    (args.out / "tagger_config.json").write_text(json.dumps(config, indent=2) + "\n")
    shutil.copy(args.model / "tokenizer.json", args.out / "tokenizer.json")
    for path in sorted(args.out.iterdir()):
        print(f"{path.stat().st_size / 1e6:8.1f} MB  {path.name}")


if __name__ == "__main__":
    main()
