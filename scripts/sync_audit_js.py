"""Regenerate the LEXICON in web/audit.js from chess_tagger/audit.py (run after editing the Python lexicon)."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from chess_tagger.audit import LEXICON  # noqa: E402

path = ROOT / "web" / "audit.js"
js = path.read_text()
js = re.sub(r"export const LEXICON = \{.*?\n\};", lambda _: "export const LEXICON = " + json.dumps(LEXICON, indent=2) + ";", js, flags=re.S)  # a function: no backslash processing
path.write_text(js)
print(f"{sum(len(v) for v in LEXICON.values())} patterns -> {path}")
