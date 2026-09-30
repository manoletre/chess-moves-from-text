// The chess move & board-location tagger, running in the browser.
// Mirrors chess_tagger/model.py: same tokenizer, same labels, same decoding rules.

const MOVE_TYPES = new Set(["START", "CONT"]);
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

// One predicted label per token -> [[start, end, type], ...] character spans.
export function decode(text, offsets, labelIds, labels) {
  const spans = [];
  let insideSpan = false;
  offsets.forEach(([start, end], i) => {
    while (start < end && /\s/.test(text[start])) start++;
    const label = labels[labelIds[i]];
    if (start === end || label === "O") { insideSpan = false; return; }
    const [tag, kind] = label.split("-");
    if (tag === "B" || !insideSpan) spans.push([start, end, kind]);
    else spans[spans.length - 1][1] = end;
    insideSpan = true;
  });
  const cleaned = [];
  for (let [start, end, kind] of spans) {
    while (start < end && TRIM.has(text[start])) start++;
    while (end > start && TRIM.has(text[end - 1])) end--;
    if (start < end) cleaned.push([start, end, kind]);
  }
  return cleaned;
}

// Spans plus the two views the app needs: move series and board highlights (same shape as ChessTagger.tag).
export function group(text, spans) {
  const out = [], series = [], highlights = [];
  for (const [start, end, type] of spans) {
    const item = { start, end, type, text: text.slice(start, end) };
    if (MOVE_TYPES.has(type)) {
      if (type === "START" || !series.length) series.push([]);
      item.series = series.length - 1;
      item.index = series.at(-1).length;
      series.at(-1).push(item);
    } else {
      highlights.push(item);
    }
    out.push(item);
  }
  return { spans: out, series, highlights };
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
  const labels = config.labels;

  return async function tag(text) {
    const { ids, offsets, truncated } = tokenizer.encode(text, config.max_length);
    if (ids.length < 2) return { ...group(text, []), truncated };
    const input = new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, ids.length]);
    const { logits } = await session.run({ input_ids: input });
    const n = labels.length;
    const labelIds = [];
    for (let t = 0; t < ids.length; t++) {
      let best = 0;
      for (let k = 1; k < n; k++) if (logits.data[t * n + k] > logits.data[t * n + best]) best = k;
      labelIds.push(best);
    }
    return { ...group(text, decode(text, offsets, labelIds, labels)), truncated };
  };
}
