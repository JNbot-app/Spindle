"""The BPE tokenizer stored inside every .cact archive (pure Python).

Vocabulary, scores, byte fallback and special tokens all come from the archive.
"""
import struct
import math
import heapq
import re
from numbers import Integral

TK_NORMAL, TK_UNKNOWN, TK_CONTROL, TK_USER_DEFINED, TK_BYTE = 0, 1, 2, 3, 4
_TK_HDR = "<IIIIIBBH"
_TK_REC = "<fBH"

_SP_META_SPACE = "▁"


def parse_tokenizer_blob(blob):
    off = struct.calcsize(_TK_HDR)
    if len(blob) < off:
        raise ValueError("Truncated tokenizer header")
    n, pad, eos, bos, unk, add_dummy, byte_fb, _ = struct.unpack_from(_TK_HDR, blob, 0)
    if not 1 <= n <= 100000 or max(pad,eos,bos,unk) >= n or add_dummy > 1 or byte_fb > 1:
        raise ValueError("Invalid tokenizer metadata")
    rec = struct.calcsize(_TK_REC)
    pieces, scores, types = [], [], []
    for _ in range(n):
        if off + rec > len(blob):
            raise ValueError("Truncated tokenizer record")
        score, t, ln = struct.unpack_from(_TK_REC, blob, off)
        off += rec
        if not math.isfinite(score) or t > TK_BYTE or not ln or off + ln > len(blob):
            raise ValueError("Invalid tokenizer record")
        pieces.append(blob[off:off + ln].decode("utf-8"))
        scores.append(score)
        types.append(t)
        off += ln
    if off != len(blob) or len(set(pieces)) != n:
        raise ValueError("Invalid tokenizer vocabulary")
    return {"pieces": pieces, "scores": scores, "types": types, "pad_id": pad,
            "eos_id": eos, "bos_id": bos, "unk_id": unk,
            "add_dummy_prefix": bool(add_dummy), "byte_fallback": bool(byte_fb)}


class Tokenizer:
    def __init__(self, meta):
        self.meta = meta
        self.pieces = meta["pieces"]
        self.scores = meta["scores"]
        self.types = meta["types"]
        self.add_dummy = meta["add_dummy_prefix"]
        self.byte_fallback = meta["byte_fallback"]
        self.unk_id = meta["unk_id"]
        self.p2id = {p: i for i, p in enumerate(self.pieces)}
        self.byte_id = {int(p[3:5], 16): i
                        for i, (p, t) in enumerate(zip(self.pieces, self.types)) if t == TK_BYTE}
        self.markers = sorted((p for p, t in zip(self.pieces, self.types) if t == TK_USER_DEFINED),
                              key=len, reverse=True)
        self._marker_pattern=re.compile('|'.join(re.escape(p) for p in self.markers)) if self.markers else None

    @classmethod
    def from_archive(cls, path):
        from .archive import Archive
        with Archive(path) as archive:
            return cls(parse_tokenizer_blob(archive.tokenizer_blob()))

    @property
    def bos_id(self):
        return self.meta["bos_id"]

    @property
    def eos_id(self):
        return self.meta["eos_id"]

    @property
    def pad_id(self):
        return self.meta["pad_id"]

    def _bpe(self, seg, offsets=False):
        syms = list(seg)
        size=len(syms)
        previous=list(range(-1,size-1))
        following=list(range(1,size))+[-1] if size else []
        versions=[0]*size
        candidates=[]
        def offer(left):
            if left<0 or not syms[left]: return
            right=following[left]
            if right<0: return
            idx=self.p2id.get(syms[left]+syms[right])
            if idx is not None:
                # Original character positions retain left-to-right tie order.
                heapq.heappush(candidates,(-self.scores[idx],left,right,versions[left],versions[right]))
        for left in range(size-1): offer(left)
        while candidates:
            _,left,right,left_version,right_version=heapq.heappop(candidates)
            if (not syms[left] or not syms[right] or following[left]!=right or
                    versions[left]!=left_version or versions[right]!=right_version):
                continue
            syms[left]+=syms[right]
            syms[right]=''
            versions[left]+=1
            versions[right]+=1
            following[left]=following[right]
            if following[right]>=0: previous[following[right]]=left
            offer(previous[left])
            offer(left)
        ids = []
        for start, s in enumerate(syms):
            if not s: continue
            idx = self.p2id.get(s)
            if idx is not None:
                ids.append((idx, start, start+len(s)) if offsets else idx)
            elif self.byte_fallback:
                if offsets:
                    ids.extend((self.byte_id[b], start+j, start+j+1)
                               for j, char in enumerate(s) for b in char.encode('utf-8'))
                else:
                    ids.extend(self.byte_id[b] for b in s.encode("utf-8"))
            else:
                ids.append((self.unk_id, start, start+len(s)) if offsets else self.unk_id)
        return ids

    def encode(self, text):
        return self._encode(text, offsets=False)

    def encode_with_offsets(self, text):
        """Return (token id, character start, character end), matching encode exactly."""
        return self._encode(text, offsets=True)

    def _encode(self, text, offsets):
        if not isinstance(text,str):
            raise TypeError("Tokenizer input must be text")
        if not text:
            return []
        esc = text.replace(" ", _SP_META_SPACE)
        if self.add_dummy:
            esc = _SP_META_SPACE + esc
        ids=[]
        position=0
        def append_segment(start, end):
            encoded = self._bpe(esc[start:end], offsets)
            if offsets:
                ids.extend((token, a+start, b+start) for token, a, b in encoded)
            else:
                ids.extend(encoded)
        if self._marker_pattern:
            for match in self._marker_pattern.finditer(esc):
                append_segment(position, match.start())
                token = self.p2id[match.group()]
                ids.append((token, match.start(), match.end()) if offsets else token)
                position=match.end()
        append_segment(position, len(esc))
        if offsets and self.add_dummy:
            return [(token, max(0, a-1), max(0, b-1)) for token, a, b in ids]
        return ids

    def decode(self, ids):
        buf = bytearray()
        for i in ids:
            if not isinstance(i, Integral) or isinstance(i, bool) or not 0 <= i < len(self.pieces):
                raise ValueError("Token ID outside the vocabulary")
            t = self.types[i]
            if t == TK_BYTE:
                buf.append(int(self.pieces[i][3:5], 16))
            elif t in (TK_CONTROL, TK_UNKNOWN):
                continue
            else:
                buf += self.pieces[i].encode("utf-8")
        text = buf.decode("utf-8", "replace").replace(_SP_META_SPACE, " ")
        if self.add_dummy and text.startswith(" "):
            text = text[1:]
        return text
