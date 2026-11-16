// The BPE tokenizer stored inside every .cact archive (port of spindle/tokenizer.py).

const TK_UNKNOWN = 1, TK_CONTROL = 2, TK_USER_DEFINED = 3, TK_BYTE = 4;
const SPACE = '▁';
const ARCHIVE_TAG = 0x05E12A84;
const HEADER_BYTES = 196, RECORD_BYTES = 44, RAW = 4;

/** Read the model geometry and the tokenizer blob out of a .cact archive (an ArrayBuffer). */
export function readArchive(buffer) {
  const view = new DataView(buffer);
  if (view.getUint32(0, true) !== ARCHIVE_TAG) throw new Error('Not a model archive');
  const count = view.getUint32(4, true), books = view.getUint32(8, true);
  const vocab = view.getUint32(6 * 4, true) || view.getUint32(5 * 4, true);
  const context = view.getUint32(13 * 4, true);
  const start = HEADER_BYTES + books * 4;
  const raws = [];
  for (let i = 0; i < count; i++) {
    const at = start + i * RECORD_BYTES;
    if (view.getUint8(at) === RAW) {
      raws.push([Number(view.getBigUint64(at + 20, true)), Number(view.getBigUint64(at + 28, true))]);
    }
  }
  if (raws.length !== 1) throw new Error('Expected exactly one tokenizer in the archive');
  const [offset, size] = raws[0];
  return { vocab, context, tokenizerBlob: new Uint8Array(buffer, offset, size).slice() };
}

export function parseTokenizerBlob(blob) {
  const view = new DataView(blob.buffer, blob.byteOffset, blob.byteLength);
  const [n, pad, eos, bos, unk] = [0, 1, 2, 3, 4].map((i) => view.getUint32(i * 4, true));
  const addDummy = view.getUint8(20) === 1, byteFallback = view.getUint8(21) === 1;
  let offset = 24;
  const decoder = new TextDecoder();
  const pieces = [], scores = [], types = [];
  for (let i = 0; i < n; i++) {
    const score = view.getFloat32(offset, true), type = view.getUint8(offset + 4), length = view.getUint16(offset + 5, true);
    offset += 7;
    pieces.push(decoder.decode(blob.subarray(offset, offset + length)));
    scores.push(score);
    types.push(type);
    offset += length;
  }
  return { pieces, scores, types, padId: pad, eosId: eos, bosId: bos, unkId: unk, addDummy, byteFallback };
}

class Heap {
  constructor() { this.items = []; }
  get size() { return this.items.length; }
  static less(a, b) {
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return a[i] < b[i];
    return false;
  }
  push(item) {
    const items = this.items;
    items.push(item);
    let i = items.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (!Heap.less(items[i], items[parent])) break;
      [items[i], items[parent]] = [items[parent], items[i]];
      i = parent;
    }
  }
  pop() {
    const items = this.items, top = items[0], last = items.pop();
    if (items.length) {
      items[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1, r = l + 1;
        let m = i;
        if (l < items.length && Heap.less(items[l], items[m])) m = l;
        if (r < items.length && Heap.less(items[r], items[m])) m = r;
        if (m === i) break;
        [items[i], items[m]] = [items[m], items[i]];
        i = m;
      }
    }
    return top;
  }
}

export class Tokenizer {
  constructor(meta) {
    Object.assign(this, meta);
    this.p2id = new Map(meta.pieces.map((p, i) => [p, i]));
    this.byteId = new Map();
    meta.pieces.forEach((p, i) => { if (meta.types[i] === TK_BYTE) this.byteId.set(parseInt(p.slice(3, 5), 16), i); });
    const markers = meta.pieces.filter((_, i) => meta.types[i] === TK_USER_DEFINED).sort((a, b) => b.length - a.length);
    this.markerPattern = markers.length
      ? new RegExp(markers.map((m) => m.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|'), 'g') : null;
    this.encoder = new TextEncoder();
  }

  _bpe(segment) {
    const syms = Array.from(segment), size = syms.length;
    const previous = Array.from({ length: size }, (_, i) => i - 1);
    const following = Array.from({ length: size }, (_, i) => (i + 1 < size ? i + 1 : -1));
    const versions = new Array(size).fill(0);
    const heap = new Heap();
    const offer = (left) => {
      if (left < 0 || !syms[left]) return;
      const right = following[left];
      if (right < 0) return;
      const id = this.p2id.get(syms[left] + syms[right]);
      if (id !== undefined) heap.push([-this.scores[id], left, right, versions[left], versions[right]]);
    };
    for (let left = 0; left < size - 1; left++) offer(left);
    while (heap.size) {
      const [, left, right, leftVersion, rightVersion] = heap.pop();
      if (!syms[left] || !syms[right] || following[left] !== right ||
          versions[left] !== leftVersion || versions[right] !== rightVersion) continue;
      syms[left] += syms[right];
      syms[right] = '';
      versions[left]++;
      versions[right]++;
      following[left] = following[right];
      if (following[right] >= 0) previous[following[right]] = left;
      offer(previous[left]);
      offer(left);
    }
    const ids = [];
    for (const s of syms) {
      if (!s) continue;
      const id = this.p2id.get(s);
      if (id !== undefined) ids.push(id);
      else if (this.byteFallback) for (const b of this.encoder.encode(s)) ids.push(this.byteId.get(b));
      else ids.push(this.unkId);
    }
    return ids;
  }

  encode(text) {
    if (!text) return [];
    let escaped = text.replaceAll(' ', SPACE);
    if (this.addDummy) escaped = SPACE + escaped;
    const ids = [];
    let position = 0;
    if (this.markerPattern) {
      for (const match of escaped.matchAll(this.markerPattern)) {
        ids.push(...this._bpe(escaped.slice(position, match.index)), this.p2id.get(match[0]));
        position = match.index + match[0].length;
      }
    }
    ids.push(...this._bpe(escaped.slice(position)));
    return ids;
  }

  /** Raw bytes of each token, as the grammar sees them. */
  pieceBytes() {
    return this.pieces.map((p, i) => (this.types[i] === TK_BYTE
      ? Uint8Array.of(parseInt(p.slice(3, 5), 16))
      : this.encoder.encode(p.replaceAll(SPACE, ' '))));
  }

  isSpecial(id) { return this.types[id] === TK_UNKNOWN || this.types[id] === TK_CONTROL; }
}
