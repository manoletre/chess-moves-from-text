// The chess tagger (v3: moves, pieces, locations and claims), running in the browser.
// Mirrors chess_tagger/model.py: same tokenizer, same heads, same decoding rules.

const TRIM = new Set(" \n\t.,;:!?*`()\"'"); // never part of a move or location (but + # = - are)

// ------------------------------------------------------------ tokenizer
// Byte-level BPE as described by tokenizer.json (no normalizer), plus the
// character offsets of every token, which transformers.js doesn't expose.

// tokenizer.json's split regex, with (?i:'s|'t|...) spelled out: JS has no inline flags everywhere yet.
const SPLIT = /'(?:[sS]|[tT]|[rR][eE]|[vV][eE]|[mM]|[lL][lL]|[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;

function byteToUnicode() {
  const bytes = [];
  for (let b = 33; b <= 126; b++) bytes.push(b);
  for (let b = 161; b <= 172; b++) bytes.push(b);
  for (let b = 174; b <= 255; b++) bytes.push(b);
  const chars = [...bytes];
  for (let b = 0, n = 0; b < 256; b++) {
    if (!bytes.includes(b)) { bytes.push(b); chars.push(256 + n++); }
  }
  const table = new Array(256);
  bytes.forEach((b, i) => { table[b] = String.fromCharCode(chars[i]); });
  return table;
}

export class Tokenizer {
  constructor(json) {
    this.vocab = new Map(Object.entries(json.model.vocab));
    this.ranks = new Map(json.model.merges.map((m, i) => [Array.isArray(m) ? m.join(" ") : m, i]));
    this.bos = json.added_tokens.find((t) => t.content === "<|startoftext|>").id;
    this.bytes = byteToUnicode();
    this.encoder = new TextEncoder();
    this.cache = new Map();
  }

  // One pre-tokenized piece -> BPE tokens (as byte-level strings).
  bpe(piece) {
    if (this.cache.has(piece)) return this.cache.get(piece);
    let parts = Array.from(this.encoder.encode(piece), (b) => this.bytes[b]);
    while (parts.length > 1) {
      let best = -1, bestRank = Infinity;
      for (let i = 0; i < parts.length - 1; i++) {
        const rank = this.ranks.get(parts[i] + " " + parts[i + 1]);
        if (rank !== undefined && rank < bestRank) { bestRank = rank; best = i; }
      }
      if (best < 0) break;
      const pair = parts[best] + " " + parts[best + 1];
      const merged = [];
      for (let i = 0; i < parts.length; i++) {
        if (i < parts.length - 1 && parts[i] + " " + parts[i + 1] === pair) { merged.push(parts[i] + parts[i + 1]); i++; }
        else merged.push(parts[i]);
      }
      parts = merged;
    }
    this.cache.set(piece, parts);
    return parts;
  }

  // text -> { ids, offsets } with [start, end) offsets in JS string indices; ids[0] is BOS.
  encode(text, maxLength = Infinity) {
    const ids = [this.bos];
    const offsets = [[0, 0]];
    for (const match of text.matchAll(SPLIT)) {
      const piece = match[0];
      // For every UTF-8 byte: where its character starts and ends in the string.
      // A token that splits a character gets the whole character.
      const charStart = [], charEnd = [];
      let index = match.index;
      for (const ch of piece) {
        for (let k = this.encoder.encode(ch).length; k > 0; k--) {
          charStart.push(index);
          charEnd.push(index + ch.length);
        }
        index += ch.length;
      }
      let byte = 0;
      for (const token of this.bpe(piece)) {
        const start = charStart[byte];
        byte += token.length; // one character per byte in the byte-level alphabet
        ids.push(this.vocab.get(token));
        offsets.push([start, charEnd[byte - 1]]);
      }
    }
    return { ids: ids.slice(0, maxLength), offsets: offsets.slice(0, maxLength), truncated: ids.length > maxLength };
  }
}

// --------------------------------------------------------------- decoding
// The model has one output layer split into heads (tagger_config.json "heads", in order):
// mention (BIO spans), line + stance (read on MOVE spans), frame (read on PIECE spans and claims),
// claim (BIO spans). Mirrors chess_tagger/model.py.

function headSlices(heads) {
  const out = {};
  let offset = 0;
  for (const [name, labels] of Object.entries(heads)) {
    out[name] = { offset, labels };
    offset += labels.length;
  }
  return { slices: out, width: offset };
}

// One BIO label per token -> [[start, end, type, firstToken, lastToken], ...].
function bioSpans(text, offsets, labels) {
  const spans = [];
  let inside = false;
  offsets.forEach(([start, end], t) => {
    while (start < end && /\s/.test(text[start])) start++;
    const label = labels[t];
    if (start === end || label === "O") { inside = false; return; }
    const [tag, kind] = label.split("-");
    const last = spans.at(-1);
    const midWord = last && last[1] === start && start > 0 && /[\p{L}\p{N}]/u.test(text[start - 1]) && /[\p{L}\p{N}]/u.test(text[start]);
    // a word is never split between two spans: the first token decides
    if (midWord || (inside && tag === "I" && kind === last[2])) { last[1] = end; last[4] = t; }
    else spans.push([start, end, kind, t, t]);
    inside = true;
  });
  const cleaned = [];
  const alnum = (c) => c !== undefined && /[\p{L}\p{N}]/u.test(c);
  for (let [start, end, kind, t0, t1] of spans) {
    // a span never starts or ends in the middle of a word ("ra|kes")
    while (start > 0 && start < end && alnum(text[start - 1]) && alnum(text[start])) start--;
    while (end < text.length && end > start && alnum(text[end]) && alnum(text[end - 1])) end++;
    while (start < end && TRIM.has(text[start])) start++;
    while (end > start && TRIM.has(text[end - 1])) end--;
    if (start < end) cleaned.push([start, end, kind, t0, t1]);
  }
  return cleaned;
}

// logits: Float32Array of [tokens x width] -> { mentions: [[s, e, type, line, stance, frame]], claims: [[s, e, type, frame]] }
export function decode(text, offsets, logits, heads) {
  const { slices, width } = headSlices(heads);
  const at = (t, k) => logits[t * width + k];
  const argmaxLabels = ({ offset, labels }) => offsets.map((_, t) => {
    let best = 0;
    for (let k = 1; k < labels.length; k++) if (at(t, offset + k) > at(t, offset + best)) best = k;
    return labels[best];
  });
  // average the head's log-probabilities over the span's tokens
  const attribute = ({ offset, labels }, t0, t1) => {
    const sums = new Array(labels.length).fill(0);
    for (let t = t0; t <= t1; t++) {
      const row = labels.map((_, k) => at(t, offset + k));
      const max = Math.max(...row);
      const logZ = max + Math.log(row.reduce((acc, v) => acc + Math.exp(v - max), 0));
      row.forEach((v, k) => { sums[k] += v - logZ; });
    }
    return labels[sums.indexOf(Math.max(...sums))];
  };
  const mentions = bioSpans(text, offsets, argmaxLabels(slices.mention)).map(([s, e, kind, t0, t1]) => [
    s, e, kind,
    kind === "MOVE" ? attribute(slices.line, t0, t1) : null,
    kind === "MOVE" ? attribute(slices.stance, t0, t1) : null,
    kind === "PIECE" ? attribute(slices.frame, t0, t1) : null,
  ]);
  const firstMove = mentions.find((m) => m[2] === "MOVE");
  if (firstMove) firstMove[3] = "NEW"; // nothing to continue yet
  const claims = bioSpans(text, offsets, argmaxLabels(slices.claim))
    .map(([s, e, kind, t0, t1]) => [s, e, kind, attribute(slices.frame, t0, t1)]);
  return { mentions, claims };
}

// The views the app needs (same shape as chess_tagger.model.group): every span in text order,
// move series (NEW starts one, NEXT extends it, ALT replaces its last move) and board highlights.
export function group(text, { mentions, claims }) {
  const series = [], highlights = [], spans = [];
  for (const [start, end, type, line, stance, frame] of mentions) {
    const item = { start, end, type, kind: "mention", text: text.slice(start, end) };
    if (type === "MOVE") {
      if (line === "NEW" || !series.length) series.push([]);
      else if (line === "ALT" && series.at(-1).length) series.push(series.at(-1).slice(0, -1));
      Object.assign(item, { line, stance, series: series.length - 1, index: series.at(-1).length });
      series.at(-1).push(item);
    } else {
      if (type === "PIECE") item.frame = frame;
      highlights.push(item);
    }
    spans.push(item);
  }
  for (const [start, end, type, frame] of claims) {
    spans.push({ start, end, type, frame, kind: "claim", text: text.slice(start, end) });
  }
  spans.sort((a, b) => a.start - b.start || a.end - b.end);
  // the heads are independent, so a claim can overlap a mention: keep the first
  const kept = spans.filter((s, i) => !spans.slice(0, i).some((o) => o.end > s.start && o.start < s.end));
  return { spans: kept, series, highlights };
}

// -------------------------------------------------------------- the model

// Download with progress, and keep a copy in the Cache API so the next visit is instant.
async function fetchCached(url, cache, onBytes) {
  const cached = await cache?.match(url);
  if (cached) {
    const bytes = new Uint8Array(await cached.arrayBuffer());
    onBytes(bytes.length);
    return bytes;
  }
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  const reader = response.body.getReader();
  const chunks = [];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    onBytes(value.length);
  }
  const blob = new Blob(chunks);
  await cache?.put(url, new Response(blob)).catch(() => {}); // quota errors: just don't cache
  return new Uint8Array(await blob.arrayBuffer());
}

// The model comes in parts (see scripts/export_onnx.py); download them in parallel and join them.
// cacheName: bump it when the model at baseUrl changes, or visitors keep the old one.
export async function loadTagger(ort, baseUrl, cacheName, onProgress) {
  const [config, tokenizerJson] = await Promise.all([
    fetch(`${baseUrl}/tagger_config.json`).then((r) => r.json()),
    fetch(`${baseUrl}/tokenizer.json`).then((r) => r.json()),
  ]);
  const cache = globalThis.caches ? await caches.open(cacheName) : null;
  const total = config.onnx_size;
  let loaded = 0;
  const parts = await Promise.all(config.onnx_parts.map((name) =>
    fetchCached(`${baseUrl}/${name}`, cache, (n) => { loaded += n; onProgress?.(Math.min(1, loaded / total)); })));
  const modelBytes = new Uint8Array(parts.reduce((sum, p) => sum + p.length, 0));
  parts.reduce((offset, p) => { modelBytes.set(p, offset); return offset + p.length; }, 0);

  const tokenizer = new Tokenizer(tokenizerJson);
  const session = await ort.InferenceSession.create(modelBytes, { executionProviders: ["wasm"] });
  return async function tag(text) {
    const { ids, offsets, truncated } = tokenizer.encode(text, config.max_length);
    if (ids.length < 2) return { ...group(text, { mentions: [], claims: [] }), raw: { mentions: [], claims: [] }, truncated };
    const input = new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, ids.length]);
    const { logits } = await session.run({ input_ids: input });
    const raw = decode(text, offsets, logits.data, config.heads);
    return { ...group(text, raw), raw, truncated };
  };
}
