import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs";
import { Chess } from "./vendor/chess.js";
import { loadTagger } from "./tagger.js";

const MODEL_URL = "model"; // model_q4.onnx, tokenizer.json, tagger_config.json (see scripts/export_onnx.py)
const MODEL_CACHE = "chess-move-tagger-v3";
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

const FILES = "abcdefgh";
const ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"];
const START_FEN = new Chess().fen();

const $ = (selector) => document.querySelector(selector);
const editor = $("#text");
const loader = $("#loader");

let tag = null;      // the model, once loaded
let fen = START_FEN; // the position the text talks about
let result = null;   // spans + series of the text currently shown
let pinned = null;

// ---------------------------------------------------------------- board

const board = $("#board");
const squares = {};
for (let rank = 8; rank >= 1; rank--) {
  for (let f = 0; f < 8; f++) {
    const name = FILES[f] + rank;
    const cell = document.createElement("div");
    cell.className = `sq ${(f + rank) % 2 === 1 ? "sq--dark" : "sq--light"}`; // a1 is dark
    if (rank === 1) cell.insertAdjacentHTML("beforeend", `<span class="sq__file">${FILES[f]}</span>`);
    if (f === 0) cell.insertAdjacentHTML("beforeend", `<span class="sq__rank">${rank}</span>`);
    cell.insertAdjacentHTML("beforeend", `<span class="sq__piece"></span>`);
    board.append(cell);
    squares[name] = cell;
  }
}

function drawPosition(game, { lit = [], color = null, lastMove = null, steps = {} } = {}) {
  for (const [name, cell] of Object.entries(squares)) {
    const piece = game.get(name);
    const image = piece ? `url(pieces/${piece.color}${piece.type.toUpperCase()}.svg)` : "none";
    const glyph = cell.querySelector(".sq__piece");
    if (glyph.style.backgroundImage !== image) glyph.style.backgroundImage = image;
    cell.classList.remove("is-lit", "is-from", "is-to");
    cell.style.removeProperty("--lit");
    cell.style.removeProperty("--d");
    cell.querySelector(".sq__step")?.remove();
  }
  lit.forEach((name, i) => {
    squares[name]?.classList.add("is-lit");
    squares[name]?.style.setProperty("--lit", color);
    squares[name]?.style.setProperty("--d", i);
  });
  if (lastMove) {
    squares[lastMove.from].classList.add("is-from");
    squares[lastMove.to].classList.add("is-to");
  }
  for (const [name, number] of Object.entries(steps)) {
    squares[name]?.insertAdjacentHTML("beforeend", `<span class="sq__step">${number}</span>`);
  }
}

// ------------------------------------------------- text -> board squares
// The model only finds the text; turning "c-file" into squares is the app's job.

function squaresFor(span) {
  const text = span.text.toLowerCase();
  if (span.type === "SQUARE") return /^[a-h][1-8]$/.test(text) ? [text] : [];
  if (span.type === "FILE") {
    const f = text[0];
    return FILES.includes(f) ? [1, 2, 3, 4, 5, 6, 7, 8].map((r) => f + r) : [];
  }
  if (span.type === "RANK") {
    const digit = text.match(/[1-8]/);
    const rank = digit ? Number(digit[0]) : ORDINALS.findIndex((w) => text.includes(w)) + 1;
    return rank >= 1 ? [...FILES].map((f) => f + rank) : [];
  }
  if (span.type === "DIAGONAL") {
    const m = text.match(/([a-h])([1-8])\s*[-–]\s*([a-h])([1-8])/);
    if (!m) return [];
    let f = FILES.indexOf(m[1]), r = Number(m[2]);
    const f2 = FILES.indexOf(m[3]), r2 = Number(m[4]);
    if (Math.abs(f2 - f) !== Math.abs(r2 - r)) return [m[1] + m[2], m[3] + m[4]];
    const out = [FILES[f] + r];
    while (f !== f2) { f += Math.sign(f2 - f); r += Math.sign(r2 - r); out.push(FILES[f] + r); }
    return out;
  }
  return [];
}

// Any notation -> a played move, or null. SAN/UCI/long algebraic go through chess.js; prose
// ("the queen slides to h8", "capture the pawn on d7") is matched against the legal moves,
// like chess_tagger/verify.py does (a smaller version of it).
const PIECE_LETTER = { pawn: "p", knight: "n", bishop: "b", rook: "r", queen: "q", king: "k" };
function playMove(game, text) {
  const t = text.trim().replace(/^[.…]+/, "").replace(/[!?*`]+$/, "").replace(/^0-0(-0)?/, (m) => m.replaceAll("0", "O"));
  try { return game.move(t); } catch { /* not SAN: try prose */ }
  let moves = game.moves({ verbose: true });
  const castle = t.match(/castl\w*\s*(short|long|kingside|queenside)?/i);
  if (castle) moves = moves.filter((m) => (/long|queen/i.test(castle[1] ?? "") ? /^O-O-O/ : /^O-O(?!-O)/).test(m.san));
  const two = t.match(/([a-h][1-8])\s*(?:-|x|→|->|to)\s*([a-h][1-8])/);
  const squares = t.match(/\b[a-h][1-8]\b/g) ?? [];
  if (two) moves = moves.filter((m) => m.from === two[1] && m.to === two[2]);
  else if (squares.length) {
    moves = moves.filter((m) => m.to === squares.at(-1));
    if (squares.length > 1) moves = moves.filter((m) => m.from === squares[0]);
  }
  const capture = /captur|takes?\b|taking|recaptur|grab|snap/i.test(t);
  const promo = t.match(/promot\w*\s+(?:to\s+)?(?:an?\s+)?(queen|rook|bishop|knight)/i);
  const noun = t.match(/\b(pawn|knight|bishop|rook|queen|king)s?\b/i);
  if (noun && !(promo && promo.index < noun.index)) {
    const letter = PIECE_LETTER[noun[1].toLowerCase()];
    if (/captur|takes?\b|taking|recaptur|grab|snap/i.test(t.slice(0, noun.index))) moves = moves.filter((m) => m.captured === letter);
    else moves = moves.filter((m) => m.piece === letter);
  }
  if (capture) moves = moves.filter((m) => m.captured);
  if (promo) moves = moves.filter((m) => m.promotion === PIECE_LETTER[promo[1].toLowerCase()]);
  if (!(castle || two || squares.length || noun || promo || capture) || moves.length !== 1) return null;
  return game.move(moves[0].san);
}

// Replay moves from the current position. Stops at the first move that can't be played there.
function replay(moves) {
  const game = new Chess(fen);
  const played = [];
  for (const move of moves) {
    const done = playMove(game, move.text);
    if (!done) return { game, played, illegal: move };
    played.push(done);
  }
  return { game, played, illegal: null };
}

// The board a piece or claim talks about: its frame. NOW = the current position,
// AFTER = after the last move mentioned before it (replaying that move's series).
function frameBoard(span) {
  if (span.frame !== "AFTER") return { game: new Chess(fen), played: [] };
  const before = result.spans.filter((s) => s.type === "MOVE" && s.end <= span.start).at(-1);
  if (!before) return { game: new Chess(fen), played: [] };
  return replay(result.series[before.series].slice(0, before.index + 1));
}

const DESCRIBE = {
  MOVE: (s) => `move · ${s.line.toLowerCase()} · ${s.stance.toLowerCase()}`,
  PIECE: (s) => `piece · ${s.frame.toLowerCase()}`,
};

// ------------------------------------------------ the editable text
// One contenteditable box: the user types in it, and the model's marks are drawn in it.

// Plain text of the editor and the caret position in it. Browsers may insert <br> or <div> on Enter.
function readEditor() {
  const selection = getSelection();
  const range = selection.rangeCount && editor.contains(selection.anchorNode) ? selection.getRangeAt(0) : null;
  let text = "", caret = null;
  (function walk(node) {
    node.childNodes.forEach((child, i) => {
      if (range?.endContainer === node && range.endOffset === i) caret = text.length;
      if (child.nodeType === Node.TEXT_NODE) {
        if (range?.endContainer === child) caret = text.length + range.endOffset;
        text += child.data;
      } else if (child.nodeName === "BR") {
        text += "\n";
      } else {
        if (/^(DIV|P)$/.test(child.nodeName) && text && !text.endsWith("\n")) text += "\n";
        walk(child);
      }
    });
    if (range?.endContainer === node && range.endOffset === node.childNodes.length) caret = text.length;
  })(editor);
  return { text, caret };
}

function placeCaret(offset) {
  const walker = document.createTreeWalker(editor, NodeFilter.SHOW_TEXT);
  let node;
  while ((node = walker.nextNode())) {
    if (offset <= node.data.length) {
      getSelection().collapse(node, offset);
      return;
    }
    offset -= node.data.length;
  }
  getSelection().collapse(editor, editor.childNodes.length);
}

function render(text, data) {
  const { caret } = readEditor();
  result = data;
  pinned = null;
  editor.classList.remove("has-focus");

  // Moves that can't be played from the current position get crossed out.
  const illegal = new Set();
  for (const moves of data.series) {
    const { illegal: bad } = replay(moves);
    if (bad) moves.slice(bad.index).forEach((m) => illegal.add(m));
  }

  const nodes = [];
  let cursor = 0;
  data.spans.forEach((span, i) => {
    nodes.push(document.createTextNode(text.slice(cursor, span.start)));
    const mark = document.createElement("mark");
    mark.className = `span span--${span.kind} span--${span.type}${illegal.has(span) ? " span--illegal" : ""}`
      + (span.stance ? ` span--${span.stance.toLowerCase()}` : "");
    mark.textContent = text.slice(span.start, span.end);
    mark.title = DESCRIBE[span.type]?.(span) ?? (span.kind === "claim" ? `claim · ${span.type.toLowerCase()} · ${span.frame.toLowerCase()}` : span.type.toLowerCase());
    mark.dataset.i = i;
    nodes.push(mark);
    cursor = span.end;
  });
  nodes.push(document.createTextNode(text.slice(cursor)));
  editor.replaceChildren(...nodes);
  if (caret !== null && document.activeElement === editor) placeCaret(caret);
  drawPosition(new Chess(fen));
}

function focusSpan(i) {
  const span = result.spans[i];
  editor.classList.add("has-focus");
  const lit = new Set();

  if (span.type === "MOVE") {
    const path = result.series[span.series].slice(0, span.index + 1);
    path.forEach((move) => lit.add(result.spans.indexOf(move)));
    const { game, played } = replay(path);
    const steps = {};
    played.forEach((move, k) => { steps[move.to] = k + 1; });
    drawPosition(game, { lastMove: played.at(-1), steps });
  } else if (span.type === "PIECE" || span.kind === "claim") {
    const { game, played } = frameBoard(span);
    drawPosition(game, { lastMove: played.at(-1) });
    lit.add(i);
  } else {
    drawPosition(new Chess(fen), { lit: squaresFor(span), color: `var(--${span.type})` });
    lit.add(i);
  }

  editor.querySelectorAll(".span").forEach((el) => {
    el.classList.toggle("is-lit", lit.has(Number(el.dataset.i)));
    el.classList.toggle("is-current", Number(el.dataset.i) === i);
  });
}

function blurSpan() {
  if (pinned !== null) return focusSpan(pinned);
  editor.classList.remove("has-focus");
  editor.querySelectorAll(".span").forEach((el) => el.classList.remove("is-lit", "is-current"));
  drawPosition(new Chess(fen));
}

// ------------------------------------------------------- interaction

function targetIndex(event) {
  const el = event.target.closest?.("[data-i]");
  return el && result ? Number(el.dataset.i) : null;
}

editor.addEventListener("mouseover", (e) => { const i = targetIndex(e); if (i !== null) focusSpan(i); });
editor.addEventListener("mouseout", (e) => { if (targetIndex(e) !== null) blurSpan(); });
editor.addEventListener("click", (e) => {
  const i = targetIndex(e);
  pinned = i === null || pinned === i ? null : i;
  pinned === null ? blurSpan() : focusSpan(i);
});
editor.addEventListener("paste", (e) => {
  e.preventDefault();
  document.execCommand("insertText", false, e.clipboardData.getData("text/plain"));
});

// Tag the text shown in the editor; drop the result if the user typed on in the meantime.
let running = Promise.resolve();
async function annotate() {
  if (!tag) return;
  const { text } = readEditor();
  running = running.then(async () => {
    if (readEditor().text !== text) return;
    const data = await tag(text);
    if (readEditor().text === text) render(text, data);
  });
}

let timer;
editor.addEventListener("input", (e) => {
  if (e.isComposing) return;
  editor.classList.remove("is-fresh");
  result = null;
  pinned = null;
  editor.classList.remove("has-focus");
  if (!readEditor().text.trim()) {
    fen = START_FEN; // an empty box starts over from the initial position
    fenInput.value = "";
    exampleBar.querySelectorAll(".example").forEach((c) => c.classList.remove("is-active"));
    drawPosition(new Chess(fen));
  }
  clearTimeout(timer);
  timer = setTimeout(annotate, 300);
});

// ------------------------------------------------------- start up

const examples = await (await fetch("examples.json")).json();
const exampleBar = $("#examples");

const fenInput = $("#fen");
function setFen(value) {
  try {
    fen = new Chess(value.trim() || START_FEN).fen();
    fenInput.removeAttribute("aria-invalid");
  } catch {
    fenInput.setAttribute("aria-invalid", "true");
    return false;
  }
  if (result) render(readEditor().text, result);
  else drawPosition(new Chess(fen));
  return true;
}
fenInput.addEventListener("input", () => setFen(fenInput.value));

function showExample(example, chip) {
  exampleBar.querySelectorAll(".example").forEach((c) => c.classList.toggle("is-active", c === chip));
  fen = example.fen;
  fenInput.value = example.fen;
  fenInput.removeAttribute("aria-invalid");
  result = null;
  editor.classList.add("is-fresh");
  editor.replaceChildren(example.text);
  drawPosition(new Chess(fen));
  annotate();
}

for (const example of examples) {
  const chip = document.createElement("button");
  chip.type = "button";
  chip.className = "example";
  chip.textContent = example.label;
  chip.addEventListener("click", () => showExample(example, chip));
  exampleBar.append(chip);
}
showExample(examples[0], exampleBar.firstElementChild);

try {
  tag = await loadTagger(ort, MODEL_URL, MODEL_CACHE, (fraction) => loader.style.setProperty("--progress", fraction));
  loader.hidden = true;
  annotate();
} catch (error) {
  console.error(error);
  loader.classList.add("is-error");
}
