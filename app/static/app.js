"use strict";

const EXAMPLES = [
  {
    label: "Engine report",
    text: "Stockfish prefers Black's Qxd5 over Nb8. In this short search, Nb8 loses about 1.8 pawns of evaluation. " +
      "The game continued Nb8 Nf3 c6 dxc6 Nf6 Nd5 c7. Stockfish's line after Qxd5 begins Qxd5 Ne2.\n\n" +
      "Qxd5 puts pressure on the pawn on d5, opens the c-file for the rooks, and the bishop on g2 rakes the a1-h8 diagonal toward the 7th rank.",
  },
  {
    label: "Tutor chat",
    text: "Nb8 was a mistake because it opens the position to active counterplay. The key continuation Stockfish chose was:\n\n" +
      "**Qxd5**\n(which leads to a sharp, balanced game)\n\n" +
      "This move puts pressure on Black's d5 pawn and opens lines for White to develop quickly.",
  },
  {
    label: "Square or move?",
    text: "The pawn on e4 is weak, so White should play e5! After dxe5 Nxe5 the knight eyes f7, and the e-file is suddenly open.",
  },
  {
    label: "Opening line",
    text: "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 — the main line of the Ruy Lopez. " +
      "White's bishop sits on the a4-e8 diagonal and pins nothing yet.",
  },
  {
    label: "Look-alikes",
    text: "Upload the PGN file and check your rank in the standings: you finished 3rd. " +
      "The results sheet (1-0, 0-1) is printed on A4 paper in Room B12.",
  },
];

const TYPES = ["START", "CONT", "SQUARE", "FILE", "RANK", "DIAGONAL"];
const TYPE_NAMES = {
  START: "Move from the current position",
  CONT: "Move continuing a series",
  SQUARE: "Square",
  FILE: "File",
  RANK: "Rank",
  DIAGONAL: "Diagonal",
};
const FILES = "abcdefgh";
const ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"];

const $ = (selector) => document.querySelector(selector);
const textarea = $("#text");
const button = $("#analyze");
const annotated = $("#annotated");
const readout = $("#readout");

let result = null;
let pinned = null;

// ---------------------------------------------------------------- board

const board = $("#board");
const squares = {};
for (let rank = 8; rank >= 1; rank--) {
  for (let f = 0; f < 8; f++) {
    const name = FILES[f] + rank;
    const cell = document.createElement("div");
    cell.className = `sq ${(f + rank) % 2 === 0 ? "sq--light" : "sq--dark"}`;
    if (rank === 1) cell.insertAdjacentHTML("beforeend", `<span class="sq__file">${FILES[f]}</span>`);
    if (f === 0) cell.insertAdjacentHTML("beforeend", `<span class="sq__rank">${rank}</span>`);
    board.append(cell);
    squares[name] = cell;
  }
}

function clearBoard() {
  for (const cell of Object.values(squares)) {
    cell.classList.remove("is-lit");
    cell.style.removeProperty("--lit");
    cell.style.removeProperty("--d");
    cell.querySelector(".sq__step")?.remove();
  }
}

function lightSquares(names, color) {
  names.forEach((name, i) => {
    const cell = squares[name];
    if (!cell) return;
    cell.style.setProperty("--lit", color);
    cell.style.setProperty("--d", i);
    cell.classList.add("is-lit");
  });
}

function markStep(name, number, isStart) {
  const cell = squares[name];
  if (!cell) return;
  cell.querySelector(".sq__step")?.remove();
  cell.insertAdjacentHTML("beforeend", `<span class="sq__step${isStart ? " sq__step--start" : ""}">${number}</span>`);
}

// ------------------------------------------------- text -> board squares
// The model only finds the text; turning "c-file" into squares is the app's job.

function squaresFor(span) {
  const text = span.text.toLowerCase();
  if (span.type === "SQUARE") {
    return /^[a-h][1-8]$/.test(text) ? [text] : [];
  }
  if (span.type === "FILE") {
    const f = text[0];
    return FILES.includes(f) ? [1, 2, 3, 4, 5, 6, 7, 8].map((r) => f + r) : [];
  }
  if (span.type === "RANK") {
    const digit = text.match(/[1-8]/);
    const word = ORDINALS.findIndex((w) => text.includes(w));
    const rank = digit ? Number(digit[0]) : word + 1;
    return rank >= 1 ? [...FILES].map((f) => f + rank) : [];
  }
  if (span.type === "DIAGONAL") {
    const m = text.match(/([a-h])([1-8])\s*[-–]\s*([a-h])([1-8])/);
    if (!m) return [];
    let f = FILES.indexOf(m[1]), r = Number(m[2]);
    const f2 = FILES.indexOf(m[3]), r2 = Number(m[4]);
    if (Math.abs(f2 - f) !== Math.abs(r2 - r)) return [m[1] + m[2], m[3] + m[4]];
    const df = Math.sign(f2 - f), dr = Math.sign(r2 - r);
    const out = [FILES[f] + r];
    while (f !== f2) { f += df; r += dr; out.push(FILES[f] + r); }
    return out;
  }
  return [];
}

function destination(san) {
  // Last square in the move: "exd5" -> d5, "e8=Q+" -> e8, castling -> null
  const matches = san.toLowerCase().match(/[a-h][1-8]/g);
  return matches ? matches[matches.length - 1] : null;
}

// ---------------------------------------------------------- rendering

function render(text, data) {
  result = data;
  pinned = null;
  annotated.classList.remove("has-focus");
  annotated.replaceChildren();

  let cursor = 0;
  data.spans.forEach((span, i) => {
    annotated.append(document.createTextNode(text.slice(cursor, span.start)));
    const mark = document.createElement("mark");
    mark.className = `span span--${span.type}`;
    mark.textContent = text.slice(span.start, span.end);
    mark.dataset.i = i;
    mark.tabIndex = 0;
    mark.style.setProperty("--i", i);
    mark.title = TYPE_NAMES[span.type];
    annotated.append(mark);
    cursor = span.end;
  });
  annotated.append(document.createTextNode(text.slice(cursor)));
  if (!data.spans.length) {
    annotated.insertAdjacentHTML("beforeend", `<p class="placeholder">No moves or board locations found.</p>`);
  }

  const counts = Object.fromEntries(TYPES.map((t) => [t, data.spans.filter((s) => s.type === t).length]));
  $("#legend").innerHTML = TYPES.map((t) =>
    `<li><span class="tag tag--${t}">${t}</span> <b>${counts[t]}</b></li>`).join("");

  const list = $("#series-list");
  list.replaceChildren();
  data.series.forEach((moves) => {
    const li = document.createElement("li");
    li.insertAdjacentHTML("beforeend", `<span class="origin">current position</span>`);
    moves.forEach((move) => {
      li.insertAdjacentHTML("beforeend", `<span class="arrow">→</span>`);
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip";
      chip.textContent = move.text;
      chip.dataset.i = spanIndex(move);
      li.append(chip);
    });
    list.append(li);
  });
  $("#series-block").hidden = !data.series.length;

  $("#timing").textContent = `${data.spans.length} found · ${data.elapsed_ms} ms` + (data.truncated ? " · text truncated at 512 tokens" : "");
  $("#raw").textContent = JSON.stringify(data, null, 2);
  resetReadout();
}

function resetReadout() {
  clearBoard();
  readout.style.removeProperty("--readout");
  readout.innerHTML = `<p class="readout__kicker">Board preview</p>
    <p class="readout__body">${result && result.spans.length
      ? "Hover or tap anything highlighted in the text. Click to pin it."
      : "Hover or tap anything highlighted in the text. Locations light up here; moves show the series you would replay by clicking them."}</p>`;
}

function focusSpan(i) {
  const span = result.spans[i];
  clearBoard();
  annotated.classList.add("has-focus");
  const lit = new Set();

  if (span.type === "START" || span.type === "CONT") {
    const path = result.series[span.series].slice(0, span.index + 1);
    path.forEach((move, k) => {
      lit.add(spanIndex(move));
      const dest = destination(move.text);
      if (dest) markStep(dest, k + 1, k === 0);
    });
    readout.style.setProperty("--readout", "var(--START)");
    readout.innerHTML = `<p class="readout__kicker">${span.type} · ${TYPE_NAMES[span.type]}</p>
      <p class="readout__body">Clicking <b>${escapeHtml(span.text)}</b> shows the board after:</p>
      <p class="readout__path"><span class="dim">current position</span> → ${path.map((m) => escapeHtml(m.text)).join(" → ")}</p>`;
  } else {
    const names = squaresFor(span);
    lightSquares(names, `var(--${span.type})`);
    lit.add(i);
    readout.style.setProperty("--readout", `var(--${span.type})`);
    readout.innerHTML = `<p class="readout__kicker">${TYPE_NAMES[span.type]}</p>
      <p class="readout__body">Clicking <b>${escapeHtml(span.text)}</b> highlights ${names.length === 1 ? "the square" : `${names.length} squares`}:
      <span class="readout__path">${names.join(" ")}</span></p>`;
  }

  annotated.querySelectorAll(".span").forEach((el) => {
    el.classList.toggle("is-lit", lit.has(Number(el.dataset.i)));
    el.classList.toggle("is-current", Number(el.dataset.i) === i);
  });
}

function blurSpan() {
  if (pinned !== null) return focusSpan(pinned);
  annotated.classList.remove("has-focus");
  annotated.querySelectorAll(".span").forEach((el) => el.classList.remove("is-lit", "is-current"));
  resetReadout();
}

// series[] and spans[] arrive as separate JSON objects, so match by position, not identity
function spanIndex(move) {
  return result.spans.findIndex((s) => s.start === move.start);
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ------------------------------------------------------- interaction

function targetIndex(event) {
  const el = event.target.closest("[data-i]");
  return el ? Number(el.dataset.i) : null;
}

for (const container of [annotated, $("#series-list")]) {
  container.addEventListener("mouseover", (e) => { const i = targetIndex(e); if (i !== null) focusSpan(i); });
  container.addEventListener("mouseout", (e) => { if (targetIndex(e) !== null) blurSpan(); });
  container.addEventListener("focusin", (e) => { const i = targetIndex(e); if (i !== null) focusSpan(i); });
  container.addEventListener("focusout", () => blurSpan());
  container.addEventListener("click", (e) => {
    const i = targetIndex(e);
    if (i === null) return;
    pinned = pinned === i ? null : i;
    pinned === null ? blurSpan() : focusSpan(i);
  });
}

async function analyze() {
  const text = textarea.value;
  if (!text.trim()) return;
  button.disabled = true;
  button.textContent = "Reading…";
  annotated.classList.add("is-busy");
  try {
    const response = await fetch("/api/tag", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!response.ok) throw new Error((await response.json()).detail ?? response.statusText);
    render(text, await response.json());
  } catch (error) {
    annotated.innerHTML = `<p class="placeholder">Something went wrong: ${escapeHtml(String(error.message ?? error))}</p>`;
  } finally {
    annotated.classList.remove("is-busy");
    button.disabled = false;
    button.textContent = "Annotate";
  }
}

button.addEventListener("click", analyze);
textarea.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) analyze();
});

const examples = $("#examples");
for (const example of EXAMPLES) {
  const chip = document.createElement("button");
  chip.type = "button";
  chip.className = "example";
  chip.textContent = example.label;
  chip.addEventListener("click", () => { textarea.value = example.text; analyze(); });
  examples.append(chip);
}
textarea.value = EXAMPLES[0].text;

(async function checkStatus() {
  const status = $("#status");
  try {
    const info = await (await fetch("/api/info")).json();
    status.className = "status status--ready";
    status.textContent = `ready · ${info.device}`;
    analyze();
  } catch {
    status.className = "status status--error";
    status.textContent = "server offline";
  }
})();
