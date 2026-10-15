import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs";
import { Chess } from "./vendor/chess.js";
import { loadTagger } from "./tagger.js";
import { positionBefore, verify } from "./verify.js";

const MODEL_URL = "model"; // model_q4.onnx, tokenizer.json, tagger_config.json (see scripts/export_onnx.py)
const MODEL_CACHE = "chess-move-tagger-v3.1";
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

const KIND_LABEL = {
  legal_move: "move", check: "check", mate: "mate", capture: "capture", wins: "wins material",
  attacks: "attacks / covers", flight: "king's squares", placement: "piece on square",
};
const BADGE = { true: "✓", false: "✗", unknown: "?" };
const verdictKey = (v) => (v === true ? "true" : v === false ? "false" : "unknown");
// a move carries several facts (legal, + / #, x): show the worst one
const worst = (facts) => verdictKey(facts.some((f) => f.verdict === false) ? false : facts.some((f) => f.verdict === null) ? null : true);

let report = null;   // verify() of the text shown: what chess.js says about what the tagger found
let marks = [];      // what is drawn in the text
let showCheck = true;

// The moves a move is played after, in order, ending with it (NEXT continues, ALT branches).
function pathTo(mv) {
  const path = [];
  for (let m = mv; m; m = m.parent) path.unshift(m);
  return path;
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

// What gets marked: the tagger's moves and the facts the checker reads (claims and "piece on
// square"), nothing else. Pieces and squares a fact talks about are linked on hover, not painted.
function buildMarks(text, data) {
  report = verify(fen, text, data.raw);
  const out = [];
  const factsAt = new Map();
  for (const f of report.facts) {
    if (!factsAt.has(f.start)) factsAt.set(f.start, []);
    factsAt.get(f.start).push(f);
  }
  const related = new Set(report.facts.flatMap((f) => f.related ?? []));
  for (const mv of report.moves) {
    const facts = (factsAt.get(mv.start) ?? []).filter((f) => f.end === mv.end);
    out.push({ start: mv.start, end: mv.end, kind: "move", move: mv, facts,
      cls: `mk mk--move mk--${mv.stance.toLowerCase()}`,
      title: `tagger: move · ${mv.line.toLowerCase()} · ${mv.stance.toLowerCase()}` });
  }
  for (const f of report.facts) {
    if (f.kind === "legal_move" || out.some((m) => m.kind === "move" && m.start === f.start && m.end === f.end)) continue;
    out.push({ start: f.start, end: f.end, kind: "fact", facts: [f], cls: `mk mk--fact`,
      title: `tagger: ${KIND_LABEL[f.kind]}${f.kind === "placement" ? "" : " claim"}` });
  }
  for (const m of data.raw.mentions) {
    if (m[2] === "MOVE") continue;
    const loc = ["SQUARE", "FILE", "RANK", "DIAGONAL"].includes(m[2]);
    out.push({ start: m[0], end: m[1], kind: related.has(m[0]) || !loc ? "ref" : "loc", type: m[2],
      text: text.slice(m[0], m[1]), cls: related.has(m[0]) || !loc ? "mk mk--ref" : `mk mk--loc span--${m[2]}`,
      title: m[2].toLowerCase() });
  }
  out.sort((a, b) => a.start - b.start || b.end - a.end);
  return out.filter((m, i) => !out.slice(0, i).some((o) => o.end > m.start && o.start < m.end)); // no overlaps
}

function render(text, data) {
  const { caret } = readEditor();
  result = data;
  pinned = null;
  editor.classList.remove("has-focus");
  marks = buildMarks(text, data);

  const nodes = [];
  let cursor = 0;
  marks.forEach((m, i) => {
    nodes.push(document.createTextNode(text.slice(cursor, m.start)));
    const el = document.createElement("mark");
    el.className = m.cls;
    el.textContent = text.slice(m.start, m.end);
    el.dataset.i = i;
    if (m.facts?.length) {
      el.dataset.verdict = worst(m.facts);
      el.dataset.badge = BADGE[el.dataset.verdict];
      el.title = `${m.title}\nchess.js: ${m.facts.map((f) => `${KIND_LABEL[f.kind]} ${BADGE[verdictKey(f.verdict)]} ${f.detail}`).join("; ")}`;
    } else {
      el.title = m.title;
    }
    nodes.push(el);
    cursor = m.end;
  });
  nodes.push(document.createTextNode(text.slice(cursor)));
  editor.replaceChildren(...nodes);
  if (caret !== null && document.activeElement === editor) placeCaret(caret);
  renderReport(text);
  drawPosition(new Chess(fen));
}

// ----------------------------------------------- the checker panel

const reportEl = $("#report");

function chip(mv) {
  const bad = mv.status !== "ok";
  return `<span class="chip chip--${mv.stance.toLowerCase()}${bad ? ` chip--${mv.status === "illegal" ? "illegal" : "unknown"}` : ""}" data-start="${mv.start}">${escapeHtml(mv.text)}</span>`;
}

function escapeHtml(s) {
  return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
}

function frameLabel(f) {
  if (!f.lastMove) return "now";
  return `after ${f.lastMove.san}`;
}

function renderReport(text) {
  if (!report || (!report.moves.length && !report.facts.length)) { reportEl.hidden = true; return; }
  reportEl.hidden = false;
  const counts = { true: 0, false: 0, unknown: 0 };
  report.facts.forEach((f) => counts[verdictKey(f.verdict)]++);
  const c = report.chosen;
  const decision = c
    ? `${chip(c)} <span class="col-check">${c.status === "ok" ? `<b class="v v--true">✓</b> legal: ${c.move.from}${c.move.to}` : `<b class="v v--${c.status === "illegal" ? "false" : "unknown"}">${c.status === "illegal" ? "✗" : "?"}</b> ${c.status}`}</span>`
    : `<span class="muted">no chosen move found</span>`;
  // the same line written twice ("slide the rook to h5 … So the move is Rh5#") is shown once: prose = SAN
  const merged = [];
  for (const line of report.lines) {
    const key = line.moves.map((mv) => (mv.move ? mv.move.lan : mv.text)).join(" ");
    const same = merged.find((l) => l.key === key && l.stance === line.stance);
    if (same) same.copies.push(line);
    else merged.push({ key, stance: line.stance, copies: [line] });
  }
  const lines = merged.map(({ stance, copies }) => {
    const parts = copies.map((line) => line.moves.map((mv, i) => (i === 0 ? "" : mv.line === "ALT" ? ' <span class="muted">or</span> ' : " → ") + chip(mv)).join("")).join(' <span class="muted">=</span> ');
    return `<li><span class="tag tag--${stance.toLowerCase()}">${stance.toLowerCase()}</span> ${parts}</li>`;
  }).join("");
  const rows = report.facts.filter((f) => f.kind !== "legal_move").map((f) => {
    const i = marks.findIndex((m) => m.facts?.includes(f));
    const v = verdictKey(f.verdict);
    return `<tr data-mark="${i}">
      <td><span class="kind">${KIND_LABEL[f.kind]}</span> “${escapeHtml(text.slice(f.start, f.end))}”</td>
      <td class="muted">${frameLabel(f)}</td>
      <td class="col-check"><b class="v v--${v}">${BADGE[v]}</b> ${escapeHtml(f.detail)}</td></tr>`;
  }).join("");
  reportEl.innerHTML = `
    <div class="report__head">
      <h2>What the tagger read</h2>
      <div class="toggle" role="group" aria-label="Show">
        <button type="button" data-check="0" aria-pressed="${!showCheck}">tagger only</button>
        <button type="button" data-check="1" aria-pressed="${showCheck}">+ chess.js check</button>
      </div>
    </div>
    <p class="col-check counts">chess.js: <b class="v v--true">✓ ${counts.true}</b> true · <b class="v v--false">✗ ${counts.false}</b> false · <b class="v v--unknown">? ${counts.unknown}</b> can't tell</p>
    <h3>Chosen move</h3><p>${decision}</p>
    <h3>Lines</h3><ul class="lines">${lines || '<li class="muted">none</li>'}</ul>
    <h3>Facts</h3>
    ${rows ? `<table class="facts"><thead><tr><th>tagger: claim</th><th>position</th><th class="col-check">chess.js</th></tr></thead><tbody>${rows}</tbody></table>` : '<p class="muted">no claims</p>'}`;
}

reportEl.addEventListener("click", (e) => {
  const button = e.target.closest("[data-check]");
  if (!button) return;
  showCheck = button.dataset.check === "1";
  document.body.classList.toggle("show-check", showCheck);
  reportEl.querySelectorAll("[data-check]").forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
});
reportEl.addEventListener("mouseover", (e) => {
  const row = e.target.closest("[data-mark]");
  const chipEl = e.target.closest("[data-start]");
  const i = row ? Number(row.dataset.mark) : chipEl ? marks.findIndex((m) => m.kind === "move" && m.start === Number(chipEl.dataset.start)) : -1;
  if (i >= 0) focusSpan(i);
});
reportEl.addEventListener("mouseout", (e) => { if (e.target.closest("[data-mark], [data-start]")) blurSpan(); });
document.body.classList.toggle("show-check", showCheck);

function focusSpan(i) {
  const m = marks[i];
  if (!m) return;
  editor.classList.add("has-focus");
  const lit = new Set([i]);
  const litMarks = (starts) => marks.forEach((o, k) => { if (starts.includes(o.start)) lit.add(k); });

  if (m.kind === "move") {
    const path = pathTo(m.move);
    litMarks(path.map((p) => p.start));
    const played = path.filter((p) => p.move);
    const steps = {};
    played.forEach((p, k) => { steps[p.move.to] = k + 1; });
    const last = path.at(-1);
    drawPosition(new Chess(last.after ?? last.before ?? fen), { lastMove: last.move, steps });
  } else if (m.kind === "fact") {
    const f = m.facts[0];
    litMarks(f.related ?? []);
    const color = showCheck ? `var(--v-${verdictKey(f.verdict)})` : "var(--FILE)";
    drawPosition(new Chess(f.frameFen ?? fen), { lit: f.targets ?? [], color, lastMove: f.lastMove });
  } else {
    // a location (or a piece) is shown on the position the text is at: after the last move before it
    const at = positionBefore(report, fen, m.start);
    const lit = m.kind === "loc" || m.type === "SQUARE" ? squaresFor(m) : [];
    drawPosition(new Chess(at.fen), { lit, color: `var(--${m.type ?? "SQUARE"})`, lastMove: at.lastMove });
  }
  editor.querySelectorAll(".mk").forEach((el) => {
    el.classList.toggle("is-lit", lit.has(Number(el.dataset.i)));
    el.classList.toggle("is-current", Number(el.dataset.i) === i);
  });
  reportEl.querySelectorAll("[data-mark]").forEach((row) => row.classList.toggle("is-current", Number(row.dataset.mark) === i));
}

function blurSpan() {
  if (pinned !== null) return focusSpan(pinned);
  editor.classList.remove("has-focus");
  editor.querySelectorAll(".mk").forEach((el) => el.classList.remove("is-lit", "is-current"));
  reportEl.querySelectorAll("[data-mark]").forEach((row) => row.classList.remove("is-current"));
  drawPosition(new Chess(fen));
}

// ------------------------------------------------------- interaction

function targetIndex(event) {
  const el = event.target.closest?.("[data-i]");
  return el && result && editor.contains(el) ? Number(el.dataset.i) : null;
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
