// Check a tagged text against the position: a port of chess_tagger/verify.py (same rules), so the
// demo shows what the reward checker reads. Each fact gets a verdict: true, false or null (unverifiable).
import { Chess } from "./vendor/chess.js";

const PIECES = { pawn: "p", knight: "n", bishop: "b", rook: "r", queen: "q", king: "k" };
const PIECE_WORD = /\b(pawn|knight|bishop|rook|queen|king)s?\b/i;
const NEGATION = /\b(not|no|cannot|can't|can not|never|without|nowhere|nothing|none|un\w+|longer)\b/i;
const RESTRICTIVE = /box|trap|hem|seal|block|cut off|stuck|cramp|occup|no escape|nowhere|cannot|can't|no legal/i;
const CAPTURE_VERB = /\b(captur\w*|takes?|taking|recaptur\w*|grab\w*|win\w*|snap\w*)\b/i;
const PROMOTION = /promot\w*\s+(?:to\s+)?(?:an?\s+)?(queen|rook|bishop|knight)|=([QRBN])/i;
const CASTLE = /castl\w*\s*(short|long|kingside|queenside|king-side|queen-side)?|\b([0O]-[0O](?:-[0O])?)\b/i;
const UCI = /^([a-h][1-8])([a-h][1-8])([qrbn])?$/;
const SAN = /^([KQRBNP]?[a-h]?[1-8]?x?[a-h][1-8](=?[QRBN])?[+#]?|O-O(-O)?[+#]?)$/;
const TWO_SQUARES = /([a-h][1-8])\s*(?:-|x|→|->|to)\s*([a-h][1-8])/;
const CLAUSE_END = /[.;:!?\n]|,\s*(?:but|while|and then|so|because|as|whereas)\b|\bbut\b|\bbecause\b/g;
const FILES = "abcdefgh";
const NAMES = { p: "pawn", n: "knight", b: "bishop", r: "rook", q: "queen", k: "king" };

const squareOf = (text) => (/^[a-h][1-8]$/.test(text.trim().toLowerCase()) ? text.trim().toLowerCase() : null);
const colorOf = (text) => (/white/i.test(text) ? "w" : /black/i.test(text) ? "b" : null);
const allSquares = () => [...FILES].flatMap((f) => [1, 2, 3, 4, 5, 6, 7, 8].map((r) => f + r));

// Move text in any notation -> { move, status } with status "ok", "illegal" or "ambiguous".
export function resolveMove(fen, text) {
  const game = new Chess(fen);
  const t = text.trim().replace(/…/g, "").replace(/^\.+/, "").replace(/^[*`!? ]+|[*`!? ]+$/g, "");
  const legal = game.moves({ verbose: true });
  if (!legal.length) return { move: null, status: "illegal" };
  const castle = t.match(CASTLE);
  if (castle) {
    const long = (castle[2] && castle[2].split("-").length === 3) || /^q/i.test(castle[1] ?? "");
    const m = legal.find((x) => x.san.replace(/[+#]/, "") === (long ? "O-O-O" : "O-O"));
    return m ? { move: m, status: "ok" } : { move: null, status: "illegal" };
  }
  if (UCI.test(t)) {
    const m = legal.find((x) => x.from + x.to + (x.promotion ?? "") === t);
    return m ? { move: m, status: "ok" } : { move: null, status: "illegal" };
  }
  if (SAN.test(t)) {
    // like python-chess: a missing or extra "x", "+", "#" or "=" doesn't change which move it is
    // (whether they're true is checked as separate facts)
    const loose = (san) => san.replace(/[x+#=]/g, "");
    const bare = t.replace(/[+#]$/, "");
    const m = legal.find((x) => loose(x.san) === loose(t));
    if (m) return { move: m, status: "ok" };
    // not exact: maybe an ambiguous SAN like "Rd1" with two rooks
    const to = bare.match(/([a-h][1-8])(=?[QRBN])?$/)?.[1];
    const letter = /^[KQRBN]/.test(bare) ? bare[0].toLowerCase() : "p";
    const same = legal.filter((x) => x.to === to && x.piece === letter);
    return { move: null, status: same.length > 1 ? "ambiguous" : "illegal" };
  }
  // long algebraic / arrows / prose: filter the legal moves by every constraint we can read
  let candidates = legal;
  const two = t.match(TWO_SQUARES);
  const squares = t.match(/\b[a-h][1-8]\b/g) ?? [];
  if (two) candidates = candidates.filter((x) => x.from === two[1] && x.to === two[2]);
  else if (squares.length) {
    candidates = candidates.filter((x) => x.to === squares.at(-1));
    if (squares.length > 1) candidates = candidates.filter((x) => x.from === squares[0]);
  }
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
      const rest = t.slice(first.index + first[0].length);
      const later = rest.match(PIECE_WORD);
      if (capture && later && !(promo && promo.index < first.index + first[0].length + later.index)) {
        candidates = candidates.filter((x) => x.captured === PIECES[later[1].toLowerCase()]);
      }
    }
  }
  if (capture) candidates = candidates.filter((x) => x.captured);
  if (promo) candidates = candidates.filter((x) => x.promotion === (promo[1] ? PIECES[promo[1].toLowerCase()] : promo[2].toLowerCase()));
  if (!(two || squares.length || letter || first || promo || capture)) return { move: null, status: "ambiguous" };
  if (candidates.length === 1) return { move: candidates[0], status: "ok" };
  return { move: null, status: candidates.length ? "ambiguous" : "illegal" };
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
  const c = new Chess(game.fen());
  c.remove(square);
  const king = findPiece(c, "k", piece.color);
  return !!king && c.isAttacked(king, piece.color === "w" ? "b" : "w");
}

// fen + text + raw tagger output ({ mentions: [[s, e, type, line, stance, frame]], claims: [[s, e, type, frame]] })
// -> { facts: [...], lines: [...], chosen }
export function verify(fen, text, raw) {
  const mentions = [...raw.mentions].sort((a, b) => a[0] - b[0]);
  const claims = [...raw.claims].sort((a, b) => a[0] - b[0]);
  const facts = [], moves = [];
  const defender = new Chess(fen).turn() === "w" ? "b" : "w"; // the side the puzzle is played against
  const fact = (kind, start, end, verdict, detail = "", extra = {}) =>
    facts.push({ kind, start, end, text: text.slice(start, end), verdict, detail, ...extra });

  // 1. replay the lines: NEW from the puzzle position, NEXT after the previous move, ALT next to it
  let prev = null;
  for (const m of mentions) {
    if (m[2] !== "MOVE") continue;
    const line = prev ? m[3] : "NEW";
    const before = line === "NEW" ? fen : line === "NEXT" ? prev.after : prev.before;
    let move = null, status = "unknown position", after = null;
    if (before) {
      ({ move, status } = resolveMove(before, text.slice(m[0], m[1])));
      if (move) after = move.after;
    }
    const parent = line === "NEW" ? null : line === "NEXT" ? prev : prev.parent; // the move it is played after
    const entry = { start: m[0], end: m[1], before, after, move, status, line, stance: m[4], parent, text: text.slice(m[0], m[1]) };
    moves.push(entry);
    prev = entry;
    const where = { frameFen: after ?? before, lastMove: move, related: [m[0]] };
    if (status === "ok") {
      fact("legal_move", m[0], m[1], true, move.san, { ...where, frameFen: before, lastMove: null, targets: [move.from, move.to] });
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

  const frameOf = (start, frame) => {
    if (frame === "NOW") return { fen, last: null };
    const earlier = moves.filter((mv) => mv.end <= start);
    if (!earlier.length) return { fen, last: null };
    return { fen: earlier.at(-1).after, last: earlier.at(-1) };
  };

  // 2. piece placements ("queen on h7")
  for (const m of mentions) {
    if (m[2] !== "PIECE") continue;
    const at = pieceSquare(text, m, mentions);
    if (!at?.square) continue;
    const { fen: frameFen, last } = frameOf(m[0], m[5]);
    const word = text.slice(m[0], m[1]).match(PIECE_WORD);
    const extra = { frameFen, lastMove: last?.move, related: [m[0], at.mention[0]], targets: [at.square] };
    if (!frameFen || !word) { fact("placement", m[0], at.mention[1], null, "unknown position", extra); continue; }
    const piece = new Chess(frameFen).get(at.square);
    const color = colorOf(text.slice(m[0], m[1]));
    const ok = !!piece && piece.type === PIECES[word[1].toLowerCase()] && (!color || piece.color === color);
    fact("placement", m[0], at.mention[1], ok,
      `${at.square}: ${piece ? `${piece.color === "w" ? "white" : "black"} ${NAMES[piece.type]}` : "empty"}`, extra);
  }

  // 3. claims, linked to the nearest move/piece before them in their clause and the squares/pieces after
  for (const [start, end, kind, frame] of claims) {
    const { fen: frameFen, last } = frameOf(start, frame);
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
    let objects = mentions.filter((m) => m[0] >= end && m[0] < hi && (m[2] === "SQUARE" || m[2] === "PIECE"));
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
    if (!frameFen) { fact(k, start, end, null, "unknown position", extra); continue; }
    let board = new Chess(frameFen);

    if (kind === "CHECK" || kind === "MATE") {
      const value = kind === "CHECK" ? board.isCheck() : board.isCheckmate();
      fact(k, start, end, value !== negated, value ? `${kind === "CHECK" ? "check" : "checkmate"} on the board` : `no ${kind === "CHECK" ? "check" : "mate"} on the board`, extra);
    } else if (kind === "WINS") {
      const mv = (subject && moves.find((x) => x.start === subject[0])) || last;
      if (!mv?.move) { fact(k, start, end, null, "no resolved move", extra); continue; }
      const victims = objects.filter((o) => o[2] === "PIECE").map((o) => text.slice(o[0], o[1]).match(PIECE_WORD)).filter(Boolean).map((w) => PIECES[w[1].toLowerCase()]);
      const captured = mv.move.captured ?? null;
      const ok = !!captured && (!victims.length || victims.includes(captured));
      fact(k, start, end, ok !== negated, `captures ${captured ? NAMES[captured] : "nothing"}`, { ...extra, frameFen: mv.after, lastMove: mv.move });
    } else if (kind === "ATTACKS") {
      if (!subject) { fact(k, start, end, null, "no subject", extra); continue; }
      let src = null;
      if (subject[2] === "MOVE") {
        const mv = moves.find((x) => x.start === subject[0]);
        if (mv?.move) { src = mv.move.to; board = new Chess(mv.after); extra.frameFen = mv.after; extra.lastMove = mv.move; }
      } else {
        const word = text.slice(subject[0], subject[1]).match(PIECE_WORD);
        const at = pieceSquare(text, subject, mentions);
        src = at?.square ?? (word ? findPiece(board, PIECES[word[1].toLowerCase()], colorOf(text.slice(subject[0], subject[1]))) : null);
      }
      const piece = src && board.get(src);
      if (!piece) { fact(k, start, end, null, "subject not found on the board", extra); continue; }
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
      const targets = [];
      for (const o of objects) {
        if (o[2] === "SQUARE") { const sq = squareOf(text.slice(o[0], o[1])); if (sq) targets.push(sq); }
        else {
          const w = text.slice(o[0], o[1]).match(PIECE_WORD);
          const at = pieceSquare(text, o, mentions);
          const sq = at?.square ?? (w ? findPiece(board, PIECES[w[1].toLowerCase()], colorOf(text.slice(o[0], o[1]))) : null);
          if (sq) targets.push(sq);
        }
      }
      if (!targets.length) { fact(k, start, end, null, "no target", extra); continue; }
      const unique = [...new Set(targets)];
      targets.length = 0;
      targets.push(...unique);
      const hit = targets.map((t) => board.attackers(t, piece.color).includes(src));
      extra.targets = [src, ...targets];
      fact(k, start, end, negated ? !hit.some(Boolean) : hit.every(Boolean),
        targets.map((t, i) => `${src}→${t} ${hit[i] ? "yes" : "no"}`).join(", "), extra);
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

  // the lines as the checker replays them, for the summary panel
  const lines = [];
  for (const mv of moves) {
    if (mv.line === "NEW" || !lines.length) lines.push({ stance: mv.stance, moves: [] });
    lines.at(-1).moves.push(mv);
  }
  const chosen = moves.find((mv) => mv.stance === "CHOSEN" && mv.before === fen) ?? null;
  return { facts: facts.sort((a, b) => a.start - b.start), moves, lines, chosen };
}
