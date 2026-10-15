// Check a tagged text against the position: a port of chess_tagger/verify.py (same rules), so the
// demo shows what the reward checker reads. Each fact gets a verdict: true, false or null (unverifiable).
import { Chess } from "./vendor/chess.js";

const PIECES = { pawn: "p", knight: "n", bishop: "b", rook: "r", queen: "q", king: "k" };
const PIECE_WORD = /\b(pawn|knight|bishop|rook|queen|king)s?\b/i;
// "isn't", "doesn't", "can't" ... are negations; "un-" words are, except the ones that intensify a claim
const NEGATION = /\b(not|no|cannot|can not|never|without|nowhere|nothing|none|longer|\w+n't|un(?!stoppable|avoidable|answerable|parr|stopp|deniabl|questionabl)\w+)\b/i;
const RESTRICTIVE = /box|trap|hem|seal|block|cut off|stuck|cramp|occup|no escape|nowhere|cannot|can't|no legal|helpless|caught|netted|no moves|no squares/i;
const CAPTURE_VERB = /\b(captur\w*|takes?|taking|recaptur\w*|grab\w*|win\w*|snap\w*)\b/i;
const RECAPTURE = /\b(recaptur\w*|takes? (it|back)|captur\w* (it|back)|retak\w*)\b/i;
const PROMOTION = /promot\w*\s+(?:to\s+)?(?:an?\s+)?(queen|rook|bishop|knight)|=([QRBN])/i;
const CASTLE = /castl\w*\s*(short|long|kingside|queenside|king-side|queen-side)?|\b([0O]-[0O](?:-[0O])?)\b/i;
const UCI = /^([a-h][1-8])([a-h][1-8])([qrbn])?$/;
const TWO_SQUARES = /([a-h][1-8])\s*(?:-|x|→|->|to)\s*([a-h][1-8])/;
const MATE_WORD = /mate\b/i;
const CHECK_WORD = /\bcheck/i;
const NEGATED_END = /(\bnot\b|n't|\bno\b|\bnever\b|\bwithout\b)\s+(?:\w+\s+){0,2}(?:check|mate|checkmate|over|the end|finish)/i;
const REACH = /reach|go to|move to|get to|land on|jump to/i;
const NEGATIVE_EVAL = /mistake|blunder|error|loses|losing|fails|falls short|\bbad\b|weak|wrong|refuted|dubious|too slow|inferior|goes nowhere|insufficient|harmless|toothless|premature|\bdud\b|pointless|misses|throws away|lets .{0,20}off|backfires|careless|risky|inaccura/i;
const ANAPHORA = /^\s*(that square|this square|those squares|these squares|it\b|them\b|both)/i;
const CLAUSE_END = /[.;:!?\n]|,\s*(?:but|while|and then|so|because|as|whereas)\b|\bbut\b|\bbecause\b/g;
const SENTENCE_END = /[.;!?\n]/g;
const FALLBACK = " (checked before the unplayable move)";
const FILES = "abcdefgh";
const ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"];
const NAMES = { p: "pawn", n: "knight", b: "bishop", r: "rook", q: "queen", k: "king" };

const squareOf = (text) => (/^[a-h][1-8]$/.test(text.trim().toLowerCase()) ? text.trim().toLowerCase() : null);
const colorOf = (text) => (/white/i.test(text) ? "w" : /black/i.test(text) ? "b" : null);
const allSquares = () => [...FILES].flatMap((f) => [1, 2, 3, 4, 5, 6, 7, 8].map((r) => f + r));
const pieceKind = (text) => { const w = text.match(PIECE_WORD); return w ? PIECES[w[1].toLowerCase()] : null; };

function givesMate(fen, move) {
  const g = new Chess(fen);
  g.move(move.san);
  return g.isCheckmate();
}

// Move text in any notation -> { move, status, note }. status: "ok", "illegal" or "ambiguous".
// answered: the square of the move this one replies to ("recaptures" happens there).
// note: [wrongSquare, actualSquare] when the move is clear but its starting square is wrong.
export function resolveMove(fen, text, answered = null) {
  const game = new Chess(fen);
  const t = text.trim().replace(/…/g, "").replace(/^\.+/, "").replace(/^[*`!? ]+|[*`!? ]+$/g, "");
  const legal = game.moves({ verbose: true });
  const fail = (status) => ({ move: null, status, note: null });
  if (!legal.length) return fail("illegal");
  const castle = t.match(CASTLE);
  if (castle) {
    const long = (castle[2] && castle[2].split("-").length === 3) || /^q/i.test(castle[1] ?? "");
    const m = legal.find((x) => x.san.replace(/[+#]/, "") === (long ? "O-O-O" : "O-O"));
    return m ? { move: m, status: "ok", note: null } : fail("illegal");
  }
  if (UCI.test(t.replace(/[+#]+$/, ""))) {
    const m = legal.find((x) => x.from + x.to + (x.promotion ?? "") === t.replace(/[+#]+$/, ""));
    return m ? { move: m, status: "ok", note: null } : fail("illegal");
  }
  const san = t.match(/^([KQRBN])?([a-h])?([1-8])?x?([a-h][1-8])(?:=?([QRBN]))?[+#]?$/);
  if (san) {
    // like python-chess: from-file/rank disambiguate, a missing or extra "x", "+" or "#" doesn't change
    // the move (whether they're true is checked as separate facts), a promotion must name its piece
    const [, letter, file, rank, to, promotion] = san;
    const piece = (letter ?? "p").toLowerCase();
    const found = legal.filter((x) => x.piece === piece && x.to === to && (!file || x.from[0] === file)
      && (!rank || x.from[1] === rank) && (x.promotion ?? null) === (promotion ? promotion.toLowerCase() : null));
    if (found.length === 1) return { move: found[0], status: "ok", note: null };
    return fail(found.length > 1 ? "ambiguous" : "illegal");
  }
  // long algebraic / arrows / prose: filter the legal moves by every constraint we can read
  let candidates = legal;
  const two = t.match(TWO_SQUARES);
  const squares = t.match(/\b[a-h][1-8]\b/g) ?? [];
  let fromSquare = null;
  if (two) { fromSquare = two[1]; candidates = candidates.filter((x) => x.to === two[2]); }
  else if (squares.length) {
    candidates = candidates.filter((x) => x.to === squares.at(-1));
    if (squares.length > 1) fromSquare = squares[0];
  } else if (answered && RECAPTURE.test(t)) candidates = candidates.filter((x) => x.to === answered);
  const letter = t.match(/^([KQRBN])[a-h]/);
  const first = t.match(PIECE_WORD);
  const capture = CAPTURE_VERB.test(t) || /[a-h1-8]x[a-h]/.test(t);
  const promo = t.match(PROMOTION);
  if (letter) candidates = candidates.filter((x) => x.piece === letter[1].toLowerCase());
  else if (first) {
    const kind = PIECES[first[1].toLowerCase()];
    if (CAPTURE_VERB.test(t.slice(0, first.index))) { // "capture the pawn on d7": the noun is the captured piece
      candidates = candidates.filter((x) => x.captured === kind);
    } else if (!(promo && promo.index < first.index)) {
      candidates = candidates.filter((x) => x.piece === kind);
      const restStart = first.index + first[0].length;
      const later = t.slice(restStart).match(PIECE_WORD);
      if (capture && later && !(promo && promo.index < restStart + later.index)) {
        candidates = candidates.filter((x) => x.captured === PIECES[later[1].toLowerCase()]);
      }
    }
  }
  if (capture) candidates = candidates.filter((x) => x.captured);
  if (promo) candidates = candidates.filter((x) => x.promotion === (promo[1] ? PIECES[promo[1].toLowerCase()] : promo[2].toLowerCase()));
  if (!(two || squares.length || letter || first || promo || capture)) return fail("ambiguous");
  // a check or mate said inside the move ("the queen checks", "delivers checkmate on e5") narrows it down
  if (candidates.length > 1 && MATE_WORD.test(t)) {
    const mates = candidates.filter((x) => givesMate(fen, x));
    if (mates.length) candidates = mates;
  } else if (candidates.length > 1 && CHECK_WORD.test(t)) {
    const checks = candidates.filter((x) => x.san.match(/[+#]$/));
    if (checks.length) candidates = checks;
  }
  let note = null;
  if (fromSquare) {
    const exact = candidates.filter((x) => x.from === fromSquare);
    if (!exact.length && candidates.length === 1) note = [fromSquare, candidates[0].from]; // right move, wrong square
    else candidates = exact;
  }
  if (candidates.length === 1) return { move: candidates[0], status: "ok", note };
  return fail(candidates.length ? "ambiguous" : "illegal");
}

function clauseBounds(text, pos) {
  let start = 0;
  for (const m of text.matchAll(CLAUSE_END)) {
    if (m.index + m[0].length <= pos) start = m.index + m[0].length;
    else if (m.index >= pos) return [start, m.index];
  }
  return [start, text.length];
}

// "queen on h7" -> the SQUARE mention right after a PIECE, joined by on/at/from/in.
function pieceSquare(text, mention, mentions) {
  for (const m of mentions) {
    if (m[2] === "SQUARE" && m[0] >= mention[1] && /^\s*(on|at|from|in)\s*$/.test(text.slice(mention[1], m[0]))) {
      return { square: squareOf(text.slice(m[0], m[1])), mention: m };
    }
  }
  return null;
}

// "bishop sits on the a4-e8 diagonal", "a rook on the e-file" -> the line mention after the piece.
function pieceLine(text, mention, mentions) {
  return mentions.find((m) => ["FILE", "RANK", "DIAGONAL"].includes(m[2]) && m[0] >= mention[1]
    && /^\s*(?:\w+\s+){0,2}(?:on|along|down|up)\s+the\s*$/.test(text.slice(mention[1], m[0]))) ?? null;
}

export function lineSquares(text, kind) {
  const t = text.toLowerCase();
  if (kind === "FILE") {
    const f = t.trim()[0];
    return FILES.includes(f) ? [1, 2, 3, 4, 5, 6, 7, 8].map((r) => f + r) : [];
  }
  if (kind === "RANK") {
    const digit = t.match(/[1-8]/);
    const rank = digit ? Number(digit[0]) : ORDINALS.findIndex((w) => t.includes(w)) + 1;
    return rank >= 1 ? [...FILES].map((f) => f + rank) : [];
  }
  const m = t.match(/([a-h])([1-8])\s*[-–]\s*([a-h])([1-8])/);
  if (!m) return [];
  let f = FILES.indexOf(m[1]), r = Number(m[2]);
  const f2 = FILES.indexOf(m[3]), r2 = Number(m[4]);
  if (Math.abs(f2 - f) !== Math.abs(r2 - r)) return [];
  const out = [FILES[f] + r];
  while (f !== f2) { f += Math.sign(f2 - f); r += Math.sign(r2 - r); out.push(FILES[f] + r); }
  return out;
}

function onLine(game, square, text, kind) {
  const piece = game.get(square)?.type;
  const ok = kind === "DIAGONAL" ? ["b", "q"].includes(piece) : ["r", "q"].includes(piece);
  return ok && lineSquares(text, kind).includes(square);
}

function findPiece(game, kind, color) {
  const found = allSquares().filter((sq) => {
    const p = game.get(sq);
    return p && p.type === kind && (!color || p.color === color);
  });
  return found.length === 1 ? found[0] : null;
}

function withTurn(fen, color) {
  const f = fen.split(" ");
  if (f[1] === color) return new Chess(fen);
  f[1] = color;
  f[3] = "-";
  try { return new Chess(f.join(" ")); } catch { return null; }
}

function kingMoves(fen, color) {
  const game = withTurn(fen, color);
  if (!game) return null;
  const king = findPiece(game, "k", color);
  return king ? new Set(game.moves({ square: king, verbose: true }).map((m) => m.to)) : null;
}

function isPinned(game, square) {
  const piece = game.get(square);
  if (!piece) return false;
  // pinned = the king isn't attacked through this square now, but would be with the piece gone
  const enemy = piece.color === "w" ? "b" : "w";
  const king = findPiece(game, "k", piece.color);
  if (!king) return false;
  const c = new Chess(game.fen());
  c.remove(square);
  const through = (g) => g.attackers(king, enemy).filter((sq) => ["b", "r", "q"].includes(g.get(sq).type));
  const before = new Set(through(game));
  return through(c).some((sq) => !before.has(sq));
}

// The piece that just moved, named by the square it came from ("Rxh7 ... the rook on g7 covers h8").
function movedFrom(last, square, game, kind = null) {
  if (!last?.move || !square || last.move.from !== square) return null;
  const to = last.move.to;
  return !kind || game.get(to)?.type === kind ? to : null;
}

// fen + text + raw tagger output ({ mentions: [[s, e, type, line, stance, frame]], claims: [[s, e, type, frame]] })
// -> { facts, moves, lines, chosen }
// best: the puzzle's solution move (UCI), if known. Without it, EVAL claims are unverifiable.
export function verify(fen, text, raw, best = null) {
  const mentions = [...raw.mentions].sort((a, b) => a[0] - b[0]);
  const claims = [...raw.claims].sort((a, b) => a[0] - b[0]);
  const defender = new Chess(fen).turn() === "w" ? "b" : "w"; // the side the puzzle is played against
  const facts = [], moves = [];
  let fallbackNow = false;
  const fact = (kind, start, end, verdict, detail = "", extra = {}) =>
    facts.push({ kind, start, end, text: text.slice(start, end), verdict,
      detail: detail + (fallbackNow ? FALLBACK : ""), fallback: fallbackNow, ...extra });

  // 1. replay the lines: NEW from the puzzle position, NEXT after the previous move, ALT next to it
  let prev = null;
  for (const m of mentions) {
    if (m[2] !== "MOVE") continue;
    const line = prev ? m[3] : "NEW";
    const before = line === "NEW" ? fen : line === "NEXT" ? prev.after : prev.before;
    const answered = line === "NEW" ? null : line === "NEXT" ? prev.move?.to ?? null : prev.answered;
    let move = null, status = "unknown position", after = null, note = null;
    if (before) {
      ({ move, status, note } = resolveMove(before, text.slice(m[0], m[1]), answered));
      if (move) after = move.after;
    }
    const parent = line === "NEW" ? null : line === "NEXT" ? prev : prev.parent; // the move it is played after
    const entry = { start: m[0], end: m[1], before, after, move, status, line, stance: m[4], parent, answered, text: text.slice(m[0], m[1]) };
    moves.push(entry);
    prev = entry;
    const where = { frameFen: after ?? before, lastMove: move, related: [m[0]] };
    if (status === "ok") {
      fact("legal_move", m[0], m[1], true, move.san, { ...where, frameFen: before, lastMove: null, targets: [move.from, move.to] });
      if (note) fact("placement", m[0], m[1], false, `not on ${note[0]}: the piece is on ${note[1]}`, { ...where, frameFen: before, lastMove: null, targets: note });
      const notation = text.slice(m[0], m[1]).replace(/[!?]+$/, "");
      const board = new Chess(after);
      if (notation.endsWith("#")) fact("mate", m[0], m[1], board.isCheckmate(), "# in the notation", where);
      else if (notation.endsWith("+")) fact("check", m[0], m[1], board.isCheck(), "+ in the notation", where);
      else if (/mate/i.test(notation)) fact("mate", m[0], m[1], board.isCheckmate(), "mate inside the move", where);
      else if (/\bcheck/i.test(notation)) fact("check", m[0], m[1], board.isCheck(), "check inside the move", where);
      if (/[a-h1-8]x[a-h]/.test(notation)) fact("capture", m[0], m[1], !!move.captured, "x in the notation", where);
    } else {
      fact("legal_move", m[0], m[1], status === "illegal" ? false : null, status, { frameFen: before, related: [m[0]] });
    }
  }

  // the reflection's conclusion: the last chosen move that can be played from the puzzle position
  // (a reply mistaken for a new chosen move can't be played there, so it doesn't override the decision)
  const chosenAll = moves.filter((mv) => mv.stance === "CHOSEN" && mv.before === fen);
  const chosen = [...chosenAll].reverse().find((mv) => mv.move) ?? chosenAll.at(-1) ?? null;

  // The position a fact is checked in. After an unplayable move: the last real position before it.
  const frameOf = (start, frame) => {
    if (frame === "NOW") return { fen, last: null, fallback: false };
    const earlier = moves.filter((mv) => mv.end <= start);
    if (!earlier.length) return { fen, last: null, fallback: false };
    const last = earlier.at(-1);
    if (last.after) return { fen: last.after, last, fallback: false };
    const real = [...earlier].reverse().find((mv) => mv.after && mv.end <= last.start);
    return real ? { fen: real.after, last: real, fallback: true } : { fen, last: null, fallback: true };
  };

  // 2. piece placements ("queen on h7", "the g2 pawn", "the bishop on the a4-e8 diagonal")
  for (const m of mentions) {
    if (m[2] !== "PIECE") continue;
    const kind = pieceKind(text.slice(m[0], m[1]));
    if (!kind) continue;
    const color = colorOf(text.slice(m[0], m[1]));
    let at = pieceSquare(text, m, mentions);
    if (!at?.square) {
      const before = mentions.find((s) => s[2] === "SQUARE" && s[1] <= m[0] && /^\s*$/.test(text.slice(s[1], m[0])));
      if (before) at = { square: squareOf(text.slice(before[0], before[1])), mention: before };
    }
    const { fen: frameFen, last, fallback } = frameOf(m[0], m[5]);
    fallbackNow = fallback;
    const game = new Chess(frameFen);
    if (at?.square) {
      const extra = { frameFen, lastMove: last?.move, related: [m[0], at.mention[0]], targets: [at.square] };
      const start = Math.min(m[0], at.mention[0]), end = Math.max(m[1], at.mention[1]);
      if (movedFrom(last, at.square, game, kind)) {
        fact("placement", start, end, true, `${at.square}: the ${NAMES[kind]} that just moved there`, { ...extra, targets: [last.move.to] });
      } else {
        const piece = game.get(at.square);
        const ok = !!piece && piece.type === kind && (!color || piece.color === color);
        fact("placement", start, end, ok, `${at.square}: ${piece ? `${piece.color === "w" ? "white" : "black"} ${NAMES[piece.type]}` : "empty"}`, extra);
      }
    } else {
      const line = pieceLine(text, m, mentions);
      if (line) {
        const squares = lineSquares(text.slice(line[0], line[1]), line[2]);
        const found = squares.filter((sq) => { const p = game.get(sq); return p && p.type === kind && (!color || p.color === color); });
        fact("placement", m[0], line[1], found.length > 0, `${found.length ? `on ${found[0]}` : "none"} on the ${text.slice(line[0], line[1])}`,
          { frameFen, lastMove: last?.move, related: [m[0], line[0]], targets: found.length ? found : squares });
      }
    }
    fallbackNow = false;
  }

  // 3. claims, linked to the nearest move/piece before them in their clause and the mentions after
  for (const [start, end, kind, frame] of claims) {
    const { fen: frameFen, last, fallback } = frameOf(start, frame);
    fallbackNow = fallback;
    const claimText = text.slice(start, end);
    const negated = NEGATION.test(claimText);
    let [lo, hi] = clauseBounds(text, start);
    // objects stop where the next claim's subject starts ("covers g1 and h2, and the queen covers g3")
    const passive = /^\s*by\b/.test(text.slice(end));
    const later = claims.find((x) => x[0] >= end && x[0] < hi);
    if (later && !passive) {
      const nextSubject = mentions.filter((m) => m[0] >= end && m[0] < later[0] && (m[2] === "MOVE" || m[2] === "PIECE")).at(-1);
      if (nextSubject) hi = nextSubject[0];
    }
    let subject = mentions.filter((m) => m[0] >= lo && m[0] < start && (m[2] === "MOVE" || m[2] === "PIECE")).at(-1);
    let objects = mentions.filter((m) => m[0] >= end && m[0] < hi && ["SQUARE", "PIECE", "FILE", "RANK", "DIAGONAL"].includes(m[2]));
    if (passive) { // "g1 is covered by the rook on g2": the agent is the subject, the thing before it the object
      const agent = mentions.find((m) => m[0] >= end && m[0] < (later ? later[0] : hi) && m[2] === "PIECE");
      if (agent) {
        const covered = mentions.filter((m) => m[0] >= lo && m[0] < start && (m[2] === "SQUARE" || m[2] === "PIECE")).at(-1);
        objects = covered ? [covered] : [];
        subject = agent;
      }
    }
    const extra = { frameFen, lastMove: last?.move, related: [start, ...(subject ? [subject[0]] : []), ...objects.map((o) => o[0])], targets: [] };
    const k = kind.toLowerCase();
    let board = new Chess(frameFen);

    if (kind === "CHECK" || kind === "MATE") {
      const value = kind === "CHECK" ? board.isCheck() : board.isCheckmate();
      // "there is no defence", "can't parry it" claim mate: only a negated mate/check word negates it
      const negatedEnd = NEGATED_END.test(claimText);
      fact(k, start, end, value !== negatedEnd, value ? `${kind === "CHECK" ? "check" : "checkmate"} on the board` : `no ${kind === "CHECK" ? "check" : "mate"} on the board`, extra);
    } else if (kind === "WINS") {
      const mv = (subject && moves.find((x) => x.start === subject[0])) || last;
      if (!mv?.move) { fact(k, start, end, null, "no resolved move", extra); continue; }
      const victims = objects.filter((o) => o[2] === "PIECE").map((o) => pieceKind(text.slice(o[0], o[1]))).filter(Boolean);
      const captured = mv.move.captured ?? null;
      const ok = !!captured && (!victims.length || victims.includes(captured));
      fact(k, start, end, ok !== negated, `captures ${captured ? NAMES[captured] : "nothing"}`, { ...extra, frameFen: mv.after, lastMove: mv.move });
    } else if (kind === "ATTACKS") {
      if (!subject) { fact(k, start, end, null, "no subject", extra); continue; }
      let src = null, named = null;
      if (subject[2] === "MOVE") {
        const mv = moves.find((x) => x.start === subject[0]);
        if (mv?.move) { src = mv.move.to; board = new Chess(mv.after); extra.frameFen = mv.after; extra.lastMove = mv.move; }
      } else {
        const kindS = pieceKind(text.slice(subject[0], subject[1]));
        named = pieceSquare(text, subject, mentions)?.square ?? null;
        src = movedFrom(last, named, board, kindS) ?? named ?? (kindS ? findPiece(board, kindS, colorOf(text.slice(subject[0], subject[1]))) : null);
      }
      const piece = src && board.get(src);
      if (!piece) {
        if (named) fact(k, start, end, false, `no such piece on ${named}`, extra); // a claim about a piece that isn't there
        else fact(k, start, end, null, "subject not found on the board", extra);
        continue;
      }
      extra.targets = [src];
      if (/defend|protect|guard/i.test(claimText) && /\bun|not|no\b/i.test(claimText) && !objects.length) {
        const ok = board.attackers(src, piece.color).length === 0;
        fact(k, start, end, ok, ok ? "no defenders" : "defended", extra);
        continue;
      }
      if (/pin/i.test(claimText)) { // "is pinned (to the king by the knight)": the objects only say by what
        fact(k, start, end, isPinned(board, src) !== negated, isPinned(board, src) ? "pinned" : "not pinned", extra);
        continue;
      }
      let targets = [];
      for (const o of objects) {
        if (o[2] === "SQUARE") { const sq = squareOf(text.slice(o[0], o[1])); if (sq) targets.push(sq); }
        else if (o[2] === "PIECE") {
          const kindO = pieceKind(text.slice(o[0], o[1]));
          const sq = pieceSquare(text, o, mentions)?.square ?? (kindO ? findPiece(board, kindO, colorOf(text.slice(o[0], o[1]))) : null);
          if (sq) targets.push(sq);
        }
      }
      const lines = objects.filter((o) => ["FILE", "RANK", "DIAGONAL"].includes(o[2]));
      if (!targets.length && lines.length) { // "controls the e-file": a line piece standing on that line
        const ok = lines.every((o) => onLine(board, src, text.slice(o[0], o[1]), o[2]));
        extra.targets = [src, ...lines.flatMap((o) => lineSquares(text.slice(o[0], o[1]), o[2]))];
        fact(k, start, end, ok !== negated, `${ok ? "on" : "not on"} ${lines.map((o) => text.slice(o[0], o[1])).join(", ")}`, extra);
        continue;
      }
      if (!targets.length && ANAPHORA.test(text.slice(end))) { // "covers that square / it / both": the square(s) named before
        const sentenceStart = Math.max(0, ...[...text.matchAll(SENTENCE_END)].map((m) => m.index + 1).filter((i) => i <= start));
        const earlier = mentions.filter((m) => m[2] === "SQUARE" && m[0] >= sentenceStart && m[0] < subject[0])
          .map((m) => squareOf(text.slice(m[0], m[1]))).filter((sq) => sq && sq !== src);
        targets = earlier.slice(/^\s*(both|them|those|these)/i.test(text.slice(end)) ? -2 : -1);
      }
      targets = [...new Set(targets)];
      if (!targets.length) { fact(k, start, end, null, "no target", extra); continue; }
      let hit = targets.map((t) => board.attackers(t, piece.color).includes(src));
      if (REACH.test(claimText)) hit = hit.map((h, i) => h && board.get(targets[i])?.color !== piece.color); // can't reach its own piece
      extra.targets = [src, ...targets];
      fact(k, start, end, negated ? !hit.some(Boolean) : hit.every(Boolean),
        targets.map((t, i) => `${src}→${t} ${hit[i] ? "yes" : "no"}`).join(", "), extra);
    } else if (kind === "EVAL") {
      fallbackNow = false; // an evaluation is about a move, not a position
      if (!best) { fact(k, start, end, null, "no solution move given", extra); continue; }
      let positive = !NEGATIVE_EVAL.test(claimText);
      let sideNegated = null;
      const side = text.slice(Math.max(0, start - 20), start).match(/\b(white|black)\b[\w\s']{0,12}$/i);
      if (side) { // "Black is lost", "White is winning": about a side, not a move
        const namedAttacker = (side[1].toLowerCase() === "white") === (new Chess(fen).turn() === "w");
        if (/lost|bust|hopeless|done|can'?t hold|cannot hold|resign/i.test(claimText)) {
          positive = !namedAttacker;
          sideNegated = /\bnot\b|n't (?:lost|bust|hopeless|done)/i.test(claimText);
        } else if (/win/i.test(claimText)) positive = namedAttacker;
      }
      if (sideNegated ?? negated) positive = !positive;
      const sentenceEnds = [...text.matchAll(SENTENCE_END)].map((m) => m.index);
      const sLo = Math.max(0, ...sentenceEnds.filter((i) => i + 1 <= start).map((i) => i + 1));
      const sHi = Math.min(text.length, ...sentenceEnds.filter((i) => i >= end));
      const near = moves.filter((mv) => mv.start >= sLo && mv.start < sHi);
      const beforeClaim = near.filter((mv) => mv.start < start);
      const target = beforeClaim.at(-1) ?? near[0] ?? chosen;
      if (!target?.move) { fact(k, start, end, null, "no playable move to evaluate", extra); continue; }
      const uci = target.move.from + target.move.to + (target.move.promotion ?? "");
      const isBest = uci === best;
      fact(k, start, end, isBest === positive, `${uci} ${isBest ? "is" : "is not"} the solution (${best})`,
        { ...extra, frameFen: target.before, lastMove: null, targets: [target.move.from, target.move.to], related: [start, target.start] });
    } else if (kind === "FLIGHT") {
      let color = null;
      if (subject?.[2] === "PIECE") {
        color = colorOf(text.slice(subject[0], subject[1]));
        const at = pieceSquare(text, subject, mentions);
        if (!color && at?.square && board.get(at.square)?.type === "k") color = board.get(at.square).color;
      }
      color ??= defender; // a king with no colour is the one being attacked
      const reachable = kingMoves(frameFen, color);
      if (!reachable) { fact(k, start, end, null, "no king", extra); continue; }
      const squares = objects.filter((o) => o[2] === "SQUARE").map((o) => squareOf(text.slice(o[0], o[1]))).filter(Boolean);
      const restrictive = negated || RESTRICTIVE.test(claimText);
      const ok = squares.length
        ? squares.every((s) => (restrictive ? !reachable.has(s) : reachable.has(s)))
        : restrictive ? reachable.size === 0 : reachable.size > 0;
      extra.targets = squares.length ? squares : [...reachable];
      fact(k, start, end, ok, `king can go to: ${[...reachable].sort().join(", ") || "nowhere"}`, extra);
    }
  }
  fallbackNow = false;

  // the lines as the checker replays them, for the summary panel
  const lines = [];
  for (const mv of moves) {
    if (mv.line === "NEW" || !lines.length) lines.push({ stance: mv.stance, moves: [] });
    lines.at(-1).moves.push(mv);
  }
  return { facts: facts.sort((a, b) => a.start - b.start), moves, lines, chosen };
}

// The position a text span is about, by the same text-order rule: after the last move before it.
export function positionBefore(report, fen, start) {
  const earlier = report.moves.filter((mv) => mv.end <= start && mv.after);
  return earlier.length ? { fen: earlier.at(-1).after, lastMove: earlier.at(-1).move } : { fen, lastMove: null };
}
