import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs";
import { Chess } from "./vendor/chess.js";
import { loadTagger } from "./tagger.js";

const MODEL_URL = "model"; // model_q4.onnx, tokenizer.json, tagger_config.json (see scripts/export_onnx.py)
const MODEL_CACHE = "chess-move-tagger-v2";
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

const FILES = "abcdefgh";
const ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"];
const START_FEN = new Chess().fen();
// Filled glyphs for both colours (styled in CSS); U+FE0E forces text, not emoji, rendering.
const GLYPHS = { k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟" };

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
    const glyph = cell.querySelector(".sq__piece");
    glyph.textContent = piece ? GLYPHS[piece.type] + "︎" : "";
    glyph.className = `sq__piece${piece ? ` sq__piece--${piece.color}` : ""}`;
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

// Replay moves from the current position. Stops at the first move that isn't legal there.
function replay(moves) {
  const game = new Chess(fen);
  const played = [];
  for (const move of moves) {
    try {
      played.push(game.move(move.text.replace(/^0-0(-0)?/, (m) => m.replaceAll("0", "O"))));
    } catch {
      return { game, played, illegal: move };
    }
  }
  return { game, played, illegal: null };
}

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
    mark.className = `span span--${span.type}${illegal.has(span) ? " span--illegal" : ""}`;
    mark.textContent = text.slice(span.start, span.end);
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

  if (span.type === "START" || span.type === "CONT") {
    const path = result.series[span.series].slice(0, span.index + 1);
    path.forEach((move) => lit.add(result.spans.indexOf(move)));
    const { game, played } = replay(path);
    const steps = {};
    played.forEach((move, k) => { steps[move.to] = k + 1; });
    drawPosition(game, { lastMove: played.at(-1), steps });
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
    exampleBar.querySelectorAll(".example").forEach((c) => c.classList.remove("is-active"));
    drawPosition(new Chess(fen));
  }
  clearTimeout(timer);
  timer = setTimeout(annotate, 300);
});

// ------------------------------------------------------- start up

const examples = await (await fetch("examples.json")).json();
const exampleBar = $("#examples");

function showExample(example, chip) {
  exampleBar.querySelectorAll(".example").forEach((c) => c.classList.toggle("is-active", c === chip));
  fen = example.fen;
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
