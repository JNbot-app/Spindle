"""Read and write .cact model archives (NumPy only, no JAX).

Layout (all little-endian):
  * header: 48 u32 fields and the RoPE theta (f32) - the full model geometry; fields 45-46 hold
    the decode skip mask (layers that decode tokens skip, bit i = layer i),
  * the CQ codebooks (f32): 2-bit (4) | 3-bit (8) | 4-bit (16) values, unit-group scaled,
  * a directory of 44-byte records: dtype, ndim, shape[4], offset, nbytes, group, bits
    (dtype 1 = FP16, 2 = FP32, 3 = CQ, 4 = RAW),
  * 64-byte aligned payloads in a fixed order: embedding, per layer [input norm, q, k, v,
    (q/k/v taps), q norm, k norm, gate, out, post norm, attention gate, pre-MLP norm, d1, d2, b2,
    d3, d4, w1a, w1b, w2a, w2b, w3a, w3b, cond_v, cond_u], the 9 mHC tensors, the two Hadamard
    permutations, per engram site [tables, key, value, taps], the final norm, optional heads,
    then the tokenizer (RAW).

A CQ payload for a [out, in] matrix holds the packed codes (4 or 2 bits each, one LSB-first
bitstream per row) and then one FP16 norm per group of 128 inputs. The record's `bits` field
gives the width, so one archive can hold either. `bits` = 8 marks an E8 lattice tensor (see
spindle.lattice): one uint16 code per 8 rotated values, then one FP16 gain per group.

Optional heads sit after the final norm: a manifest (FP16 codes) followed by each head's
tensors. Code 4 is the runtime LoRA: rank-r FP16 updates on the five attention projections
(q, k, v, gate, out), stored as A (layers, rank, in) and B (layers, rank, out). The engine
adds x @ A @ B to each projection, so these updates are never rounded to 4 bits.
"""
import mmap
import os
import struct
import tempfile
from pathlib import Path

import numpy as np

from spindle import lattice


def parse_layers(text):
    """'5-8,10' -> (5, 6, 7, 8, 10)."""
    layers = set()
    for part in filter(None, str(text).split(',')):
        first, _, last = part.partition('-')
        layers.update(range(int(first), int(last or first) + 1))
    return tuple(sorted(layers))

TAG = 0x05E12A84
HEADER = struct.Struct('<48If')
RECORD = struct.Struct('<BBHIIIIQQII')
FP16, FP32, CQ, RAW = 1, 2, 3, 4
ALIGN = 64
GROUP = 128

LORA_CODE = 4
LORA_TARGETS = ('q_proj', 'k_proj', 'v_proj', 'gate_proj', 'out_proj')
_ATTENTION = ('q_proj', 'k_proj', 'v_proj', 'q_taps', 'k_taps', 'v_taps', 'q_norm', 'k_norm', 'gate_proj', 'out_proj')
_HADAMARD = ('d1', 'd2', 'b2', 'd3', 'd4', 'w1a', 'w1b', 'w2a', 'w2b', 'w3a', 'w3b', 'cond_v', 'cond_u')
_FACTORS = ('w1a', 'w1b', 'w2a', 'w2b', 'w3a', 'w3b')
_MHC_SMALL = ('mhc_a_pre', 'mhc_a_post', 'mhc_a_res', 'mhc_b_pre', 'mhc_b_post', 'mhc_b_res')
_MHC_PHI = ('mhc_phi_pre', 'mhc_phi_post', 'mhc_phi_res')


def _hadamard(group):
    h = np.array([[1.0]], np.float32)
    while h.shape[0] < group:
        h = np.block([[h, h], [h, -h]])
    return (h / np.sqrt(group)).astype(np.float32)


class Archive:
    """A memory-mapped .cact archive: header, codebook, directory and tensors."""

    def __init__(self, path):
        self.path = Path(path)
        self._file = self.path.open('rb')
        self._map = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self.header = HEADER.unpack_from(self._map, 0)
        tag, count, books = self.header[:3]
        if tag != TAG or books != 28:
            self.close()
            raise ValueError(f'{path} is not a model archive')
        self.codebook = np.frombuffer(self._map, '<f4', books, HEADER.size).copy()
        start = HEADER.size + books * 4
        self.records = [RECORD.unpack_from(self._map, start + i * RECORD.size) for i in range(count)]

    @property
    def layers(self):
        return self.header[10]

    @property
    def conv_taps(self):
        return self.header[19]

    def payload(self, index):
        r = self.records[index]
        return self._map[r[7]:r[7] + r[8]]

    def tensor(self, index):
        """One tensor as FP32 (CQ matrices are dequantized), or bytes for RAW."""
        kind, ndim, _, *dims = self.records[index][:7]
        shape = tuple(dims[:ndim])
        if kind == RAW:
            return bytes(self.payload(index))
        if kind in (FP16, FP32):
            dtype = '<f2' if kind == FP16 else '<f4'
            return np.frombuffer(self.payload(index), dtype).astype(np.float32).reshape(shape)
        group, bits = self.records[index][9:11]
        if kind != CQ or ndim != 2 or group != GROUP or bits not in (2, 3, 4, lattice.E8):
            raise ValueError(f'Unsupported tensor record {index}')
        return unpack_cq(self.payload(index), shape, bits, self.codebook)

    def tokenizer_blob(self):
        raws = [i for i, r in enumerate(self.records) if r[0] == RAW]
        if len(raws) != 1:
            raise ValueError('Expected exactly one tokenizer in the archive')
        return self.tensor(raws[0])

    def close(self):
        self._map.close()
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ---------------------------------------------------------------- CQ packing

def _codebook(codebook, bits):
    if bits == lattice.E8:
        return lattice.LEVELS
    return np.asarray(codebook, np.float32)[(1 << bits) - 4:(1 << (bits + 1)) - 4]


def unpack_cq(blob, shape, bits, codebook):
    rows, columns = shape
    padded = (columns + GROUP - 1) // GROUP * GROUP
    if bits == lattice.E8:
        codes = np.frombuffer(blob, '<u2', rows * padded // 8)
        gains = np.frombuffer(blob, '<f2', rows * padded // GROUP, rows * padded // 4).astype(np.float32)
        unit = lattice.decode(codes).reshape(-1, GROUP) * gains.reshape(-1, 1)
        return (unit @ _hadamard(GROUP)).reshape(rows, padded)[:, :columns]
    packed_size = rows * padded * bits // 8
    codes = np.frombuffer(blob, np.uint8, packed_size).reshape(rows, -1)
    norms = np.frombuffer(blob, '<f2', rows * padded // GROUP, packed_size).astype(np.float32)
    if bits == 4:
        idx = np.stack([codes & 15, codes >> 4], axis=-1).reshape(rows, padded)
    else:
        bitpos = np.arange(padded) * bits
        byte, shift = bitpos // 8, bitpos % 8
        nxt = np.minimum(byte + 1, codes.shape[1] - 1)
        idx = ((codes[:, byte].astype(np.uint16) >> shift)
               | np.where(shift + bits > 8, codes[:, nxt].astype(np.uint16) << (8 - shift), 0)) & ((1 << bits) - 1)
    unit = _codebook(codebook, bits)[idx].reshape(-1, GROUP) * norms.reshape(-1, 1)
    return (unit @ _hadamard(GROUP)).reshape(rows, padded)[:, :columns]


def pack_cq(w, codebook, bits=4):
    """[out, in] FP32 matrix -> CQ payload bytes (codes, then FP16 group norms).

    `codebook` is the archive's full codebook (2 | 3 | 4-bit values); `bits` is 4, 2 or lattice.E8."""
    if bits not in (2, 4, lattice.E8):
        raise ValueError(f'CQ packing supports 2, 4 or E8 bits, not {bits}')
    w = np.asarray(w, np.float32)
    out, columns = w.shape
    pad = (-columns) % GROUP
    if pad:
        w = np.pad(w, ((0, 0), (0, pad)))
    rotated = w.reshape(out, -1, GROUP) @ _hadamard(GROUP)
    if bits == lattice.E8:
        points, gain = lattice.quantize_groups(rotated)
        return lattice.encode(points.reshape(out, -1, 8)).tobytes() + gain[..., 0].astype(np.float16).tobytes()
    norm = np.sqrt((rotated ** 2).sum(-1, keepdims=True))
    unit = (rotated / np.maximum(norm, 1e-12)).reshape(-1)
    cb = _codebook(codebook, bits)
    pos = np.clip(np.searchsorted(cb, unit), 1, len(cb) - 1)
    idx = np.where(np.abs(unit - cb[pos - 1]) <= np.abs(unit - cb[pos]), pos - 1, pos).astype(np.uint8)
    idx = idx.reshape(out, -1, 8 // bits)
    codes = np.zeros(idx.shape[:2], np.uint8)
    for k in range(8 // bits):
        codes |= idx[..., k] << (k * bits)
    return codes.tobytes() + norm[..., 0].astype(np.float16).tobytes()


# ---------------------------------------------------------------- archive <-> parameters

def model_config(header):
    """The JAX ModelConfig for an archive header."""
    from spindle.model.architecture import ModelConfig
    h = header
    orders = tuple(h[27:27 + h[26]])
    if not orders or h[22] % len(orders) or h[22] * h[21] != h[7] or h[23] != 4:
        raise ValueError('Unsupported engram geometry')
    if h[14] != 1 << (h[7] - 1).bit_length():
        raise ValueError('Unsupported Hadamard width')
    if h[31] > 13:
        raise ValueError('At most 13 engram sites (header fields 45-46 hold the decode skip mask)')
    global_mask = h[17] | h[18] << 32
    skip_mask = h[45] | h[46] << 32
    return ModelConfig(
        vocab_size=h[5], out_vocab=h[6], d_model=h[7], num_heads=h[8], num_kv_heads=h[9],
        num_layers=h[10], qk_head_dim=h[11], v_head_dim=h[12], max_seq_len=h[13],
        mhc_lanes=h[15], sliding_window=h[16],
        global_layers=tuple(i for i in range(h[10]) if global_mask >> i & 1),
        qkv_conv_taps=h[19], engram_slots=h[20], engram_sub_dim=h[21], engram_orders=orders,
        engram_heads=h[22] // len(orders), engram_seed_heads=h[25],
        engram_layers=tuple(h[32:32 + h[31]]), rope_theta=h[48],
        skip_layers=tuple(i for i in range(h[10]) if skip_mask >> i & 1), kv_window=h[3], kv_bits=h[4])


def _layer_names(taps):
    return (['ZCRMSNorm_0'] + [n for n in _ATTENTION if taps or not n.endswith('_taps')]
            + ['post_attn_norm', 'attn_gate', 'pre_hada_norm'] + list(_HADAMARD))


def load(path, bits=None):
    """(params, config, codebook, runtime_lora) for an archive.

    params use the Flax tree of SpindleModel (FP32, CQ matrices dequantized); codebook holds the
    values of the archive's weight width (16 for 4 bits, 4 for 2 bits), or of `bits` when given;
    runtime_lora is None when the archive has none.
    """
    with Archive(path) as a:
        config = model_config(a.header)
        cursor = 1
        layers = []
        for _ in range(config.num_layers):
            values = {}
            for name in _layer_names(config.qkv_conv_taps):
                values[name] = a.tensor(cursor)
                cursor += 1
            attention = {n: ({'kernel': values[n].T} if n.endswith('_proj') else
                             {'scale': values[n]} if n.endswith('_norm') else values[n])
                         for n in _ATTENTION if n in values}
            layers.append({'ZCRMSNorm_0': {'scale': values['ZCRMSNorm_0']}, 'self_attn': attention,
                           'post_attn_norm': {'scale': values['post_attn_norm']},
                           'attn_gate': values['attn_gate'].reshape(()),
                           'pre_hada_norm': {'scale': values['pre_hada_norm']},
                           'hadamard_mlp': {n: values[n] for n in _HADAMARD}})
        dims = {layers[0]['hadamard_mlp'][n].ndim for n in _FACTORS}
        if len(dims) != 1:
            raise ValueError('Archive mixes Kronecker and Monarch Hadamard factors')
        config.monarch = dims == {3}
        block = {k: _stack([layer[k] for layer in layers]) for k in layers[0]}
        stack = {'layers': {'block': block}}
        for name in _MHC_SMALL:
            stack[name] = a.tensor(cursor)
            cursor += 1
        n, L, d = config.mhc_lanes, config.num_layers, config.d_model
        for name, width in zip(_MHC_PHI, (n, n, n * n)):
            stack[name] = a.tensor(cursor).reshape(L, width, n * d).transpose(0, 2, 1)
            cursor += 1
        cursor += 2  # the Hadamard permutations are fixed (seeds 11 and 13)
        params = {'embedding': {'embedding': a.tensor(0)}, 'stack': stack}
        tables = len(config.engram_orders) * config.engram_heads
        for site in range(len(config.engram_layers)):
            params[f'engrams_{site}'] = {
                'embedding': a.tensor(cursor).reshape(tables, config.engram_slots, config.engram_sub_dim),
                'key_proj': {'kernel': a.tensor(cursor + 1).T},
                'value_proj': {'kernel': a.tensor(cursor + 2).T},
                'taps': a.tensor(cursor + 3)}
            cursor += 4
        stack['final_norm'] = {'scale': a.tensor(cursor)}
        codebook = _codebook(a.codebook, bits or a.records[2][10]).copy()
    return params, config, codebook, read_lora(path)


def _stack(values):
    if isinstance(values[0], dict):
        return {k: _stack([v[k] for v in values]) for k in values[0]}
    return np.stack(values)


def weight_bits(path):
    """Width of the archive's CQ weights: that of the first attention matrix (a 2-bit model keeps
    its embedding at 4 bits)."""
    with Archive(path) as a:
        return a.records[2][10]


def codebook(path, bits):
    """The archive's codebook values for one width (16 for 4 bits, 4 for 2 bits)."""
    with Archive(path) as a:
        return _codebook(a.codebook, bits).copy()


def backbone_tensors(params, config, codebook, bits=4):
    """(dtype, shape, payload, bits) for every backbone tensor, in archive order (no heads, no tokenizer).

    CQ matrices use `bits`; a 2-bit model keeps the embedding and the mHC mixing matrices at 4 bits
    (see quantize.keeps_4bit). bits is 0 for FP tensors."""
    def _cq(matrix, codebook, wide=False):
        matrix = np.asarray(matrix, np.float32)
        width = 4 if wide else bits
        return CQ, matrix.shape, pack_cq(matrix, codebook, width), width

    def _fp16(value):
        value = np.asarray(value, np.float16)
        return FP16, value.shape, value.tobytes(), 0

    block = params['stack']['layers']['block']
    sa, mlp = block['self_attn'], block['hadamard_mlp']
    out = [_cq(params['embedding']['embedding'], codebook, wide=True)]
    for i in range(config.num_layers):
        for name in _layer_names(config.qkv_conv_taps):
            if name.endswith('_proj'):
                out.append(_cq(np.asarray(sa[name]['kernel'][i]).T, codebook))
            elif name in ('q_norm', 'k_norm'):
                out.append(_fp16(sa[name]['scale'][i]))
            elif name.endswith('_taps'):
                out.append(_fp16(sa[name][i]))
            elif name == 'attn_gate':
                out.append(_fp16(np.asarray(block[name][i]).reshape(1)))
            elif name in _HADAMARD:
                out.append(_fp16(mlp[name][i]))
            else:
                out.append(_fp16(block[name]['scale'][i]))
    stack = params['stack']
    out += [_fp16(stack[name]) for name in _MHC_SMALL]
    for name in _MHC_PHI:
        phi = np.asarray(stack[name])
        L, nC, lanes = phi.shape
        out.append(_cq(phi.transpose(0, 2, 1).reshape(L * lanes, nC), codebook, wide=True))
    from spindle.model.architecture import hadamard_perms
    hada_n = 1 << (config.d_model - 1).bit_length()
    out += [(FP32, (hada_n,), np.asarray(p, np.float32).tobytes(), 0) for p in hadamard_perms(hada_n)]
    for site in range(len(config.engram_layers)):
        engram = params[f'engrams_{site}']
        tables = np.asarray(engram['embedding'])
        out += [_cq(tables.reshape(-1, tables.shape[-1]), codebook),
                _cq(np.asarray(engram['key_proj']['kernel']).T, codebook),
                _cq(np.asarray(engram['value_proj']['kernel']).T, codebook),
                _fp16(engram['taps'])]
    out.append(_fp16(stack['final_norm']['scale']))
    return out


def save(parent, params, config, output, runtime_lora=None, bits=None):
    """Write `params` into a copy of `parent` (same layout); heads and tokenizer are kept.

    Backbone CQ matrices are stored at `bits` (default: the parent's width).
    The parent's runtime LoRA is replaced by `runtime_lora` (or removed when it is None).
    """
    raw = Path(parent).read_bytes()
    header, start, records = _directory(raw)
    blobs = [raw[r[7]:r[7] + r[8]] for r in records]
    bits = bits or records[2][10]
    tensors = backbone_tensors(params, config, np.frombuffer(raw, '<f4', header[2], HEADER.size), bits)
    for index, ((kind, shape, blob, width), record) in enumerate(zip(tensors, records)):
        if kind != record[0] or tuple(shape) != tuple(record[3:3 + record[1]]):
            raise ValueError(f'Tensor {index} does not match the parent archive layout')
        if kind == CQ:
            record[10] = width
        blobs[index] = blob
    raw = bytearray(_rewrite(raw, header, start, records, blobs))
    mask = sum(1 << i for i in config.skip_layers)
    struct.pack_into('<II', raw, 45 * 4, mask & 0xFFFFFFFF, mask >> 32)
    raw = strip_lora(bytes(raw))
    if runtime_lora is not None:
        raw = write_lora(raw, runtime_lora)
    _atomic_write(output, raw)
    return Path(output)


def to_monarch(source, output):
    """Copy an archive with Monarch Hadamard factors: every shared factor is repeated once per group.

    The model computes exactly the same thing; training can then change each group's matrix.
    """
    raw = Path(source).read_bytes()
    header, start, records = _directory(raw)
    blobs = [raw[r[7]:r[7] + r[8]] for r in records]
    per_layer = 24 + (3 if header[19] else 0)
    first = 1 + 16 + (3 if header[19] else 0)
    for layer in range(header[10]):
        indices = [first + layer * per_layer + j for j in range(6)]
        sizes = []
        for index in indices:
            r = records[index]
            if r[0] != FP16 or r[1] != 2 or r[3] != r[4]:
                raise ValueError('Expected square FP16 Kronecker factors; is the archive already Monarch?')
            sizes.append(r[3])
        for j, index in enumerate(indices):
            copies, size = sizes[j + 1 if j % 2 == 0 else j - 1], sizes[j]
            matrix = np.frombuffer(blobs[index], np.float16).reshape(size, size)
            blobs[index] = np.ascontiguousarray(np.broadcast_to(matrix, (copies, size, size))).tobytes()
            records[index][1], records[index][3:7] = 3, [copies, size, size, 0]
    _atomic_write(output, _rewrite(raw, header, start, records, blobs))
    return Path(output)


def widen_taps(source, output, taps):
    """Copy an archive with `taps` Q/K/V convolution taps per layer; the added (older) taps are zero.

    The model computes exactly the same thing; training can then use the longer window.
    """
    raw = Path(source).read_bytes()
    header, start, records = _directory(raw)
    if not header[19] or taps < header[19]:
        raise ValueError(f'Cannot widen {header[19]} convolution taps to {taps}')
    blobs = [raw[r[7]:r[7] + r[8]] for r in records]
    names = _layer_names(header[19])
    for layer in range(header[10]):
        for name in ('q_taps', 'k_taps', 'v_taps'):
            index = 1 + layer * len(names) + names.index(name)
            r = records[index]
            if r[0] != FP16 or r[1] != 2 or r[3] != header[19]:
                raise ValueError(f'Unexpected tensor at index {index} for layer {layer} {name}')
            old = np.frombuffer(blobs[index], np.float16).reshape(r[3], r[4])
            blobs[index] = np.concatenate([old, np.zeros((taps - r[3], r[4]), np.float16)]).tobytes()
            r[3] = taps
    header = list(header)
    header[19] = taps
    _atomic_write(output, _rewrite(raw, header, start, records, blobs))
    return Path(output)


# ---------------------------------------------------------------- runtime LoRA block

def _directory(raw):
    header = HEADER.unpack_from(raw, 0)
    start = HEADER.size + header[2] * 4
    return header, start, [list(RECORD.unpack_from(raw, start + i * RECORD.size)) for i in range(header[1])]


def _rewrite(raw, header, start, records, blobs):
    position = start + len(records) * RECORD.size
    for r, blob in zip(records, blobs):
        position = (position + ALIGN - 1) & ~(ALIGN - 1)
        r[7], r[8] = position, len(blob)
        position += len(blob)
    out = bytearray(HEADER.pack(header[0], len(records), *header[2:]) + raw[HEADER.size:start])
    out += b''.join(RECORD.pack(*r) for r in records)
    for r, blob in zip(records, blobs):
        out += b'\0' * (r[7] - len(out))
        out += blob
    return bytes(out)


def _heads(header, records):
    """Index of the head manifest (or None) and of the tokenizer."""
    final_norm = 1 + header[10] * (24 + (3 if header[19] else 0)) + 11 + 4 * header[31]
    tokenizer = len(records) - 1
    if records[tokenizer][0] != RAW or records[final_norm][1] != 1:
        raise ValueError('Unexpected archive layout')
    return (final_norm + 1 if tokenizer > final_norm + 1 else None), tokenizer


def _codes(raw, record):
    return np.frombuffer(raw[record[7]:record[7] + record[8]], np.float16).astype(int).tolist()


def read_lora(path_or_bytes):
    """The runtime LoRA {target: {'A': (layers, in, rank), 'B': (layers, rank, out)}}, or None."""
    raw = path_or_bytes if isinstance(path_or_bytes, bytes) else Path(path_or_bytes).read_bytes()
    header, _, records = _directory(raw)
    manifest, tokenizer = _heads(header, records)
    if manifest is None or LORA_CODE not in _codes(raw, records[manifest]):
        return None
    first = tokenizer - 2 * len(LORA_TARGETS)
    result = {}
    for k, name in enumerate(LORA_TARGETS):
        a, b = (np.frombuffer(raw[r[7]:r[7] + r[8]], np.float16).reshape(r[3:3 + r[1]]).astype(np.float32)
                for r in records[first + 2 * k:first + 2 * k + 2])
        result[name] = {'A': a.transpose(0, 2, 1), 'B': b}
    return result


def strip_lora(raw):
    """Archive bytes without their runtime LoRA block (unchanged if there is none)."""
    header, start, records = _directory(raw)
    manifest, tokenizer = _heads(header, records)
    if manifest is None:
        return raw
    codes = _codes(raw, records[manifest])
    if LORA_CODE not in codes:
        return raw
    if codes[-1] != LORA_CODE:
        raise ValueError('The runtime LoRA must be the last head')
    blobs = [raw[r[7]:r[7] + r[8]] for r in records]
    first = tokenizer - 2 * len(LORA_TARGETS)
    records, blobs = records[:first] + records[tokenizer:], blobs[:first] + blobs[tokenizer:]
    if len(codes) == 1:
        del records[manifest], blobs[manifest]
    else:
        records[manifest][3] = len(codes) - 1
        blobs[manifest] = np.asarray(codes[:-1], np.float16).tobytes()
    return _rewrite(raw, header, start, records, blobs)


def write_lora(raw, lora):
    """Archive bytes (without a runtime LoRA) plus the given runtime LoRA block."""
    header, start, records = _directory(raw)
    manifest, tokenizer = _heads(header, records)
    blobs = [raw[r[7]:r[7] + r[8]] for r in records]
    codes = [] if manifest is None else _codes(raw, records[manifest])
    if LORA_CODE in codes:
        raise ValueError('Archive already has a runtime LoRA')
    added_records, added_blobs = [], []
    for name in LORA_TARGETS:
        for value in (np.asarray(lora[name]['A'], np.float32).transpose(0, 2, 1),
                      np.asarray(lora[name]['B'], np.float32)):
            half = value.astype(np.float16)
            if not np.isfinite(half).all():
                raise ValueError(f'Runtime LoRA exceeds the FP16 range: {name}')
            added_records.append([FP16, 3, 0, *half.shape, 0, 0, 0, 0, 0])
            added_blobs.append(half.tobytes())
    manifest_record = [FP16, 1, 0, len(codes) + 1, 0, 0, 0, 0, 0, 0, 0]
    manifest_blob = np.asarray(codes + [LORA_CODE], np.float16).tobytes()
    if manifest is None:
        records = records[:tokenizer] + [manifest_record] + added_records + records[tokenizer:]
        blobs = blobs[:tokenizer] + [manifest_blob] + added_blobs + blobs[tokenizer:]
    else:
        records[manifest], blobs[manifest] = manifest_record, manifest_blob
        records = records[:tokenizer] + added_records + records[tokenizer:]
        blobs = blobs[:tokenizer] + added_blobs + blobs[tokenizer:]
    return _rewrite(raw, header, start, records, blobs)


def _atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.spindle-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
