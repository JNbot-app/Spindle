"""ctypes binding to the native engine (native/, built as libspindle)."""
import ctypes
import json
import os
import struct
import sys
import threading
import weakref
from pathlib import Path

import numpy as np

from spindle.archive import parse_layers

ABI_VERSION = 2
_LIBRARY_NAME = {'win32': 'spindle.dll', 'darwin': 'libspindle.dylib'}.get(sys.platform, 'libspindle.so')


def find_library(library=None):
    """An explicit path, $SPINDLE_LIBRARY, or the default CMake build folder."""
    library = library or os.environ.get('SPINDLE_LIBRARY')
    if library:
        return Path(library).resolve(strict=True)
    root = Path(__file__).resolve().parents[2] / 'build' / 'native'
    for candidate in (root / _LIBRARY_NAME, root / 'Release' / _LIBRARY_NAME, root / 'bin' / _LIBRARY_NAME):
        if candidate.exists():
            return candidate
    raise FileNotFoundError('Native engine not found: build native/ (see README) or set SPINDLE_LIBRARY')


def load_library(path):
    """The native library with its argument types set (one copy per path)."""
    if path in _LIBRARIES:
        return _LIBRARIES[path]
    lib = ctypes.CDLL(str(path))
    p, i, z = ctypes.c_void_p, ctypes.c_int32, ctypes.c_size_t
    lib.np_abi_version.restype = ctypes.c_uint32
    if lib.np_abi_version() != ABI_VERSION:
        raise RuntimeError('The native engine and this Python package are out of date with each other (rebuild native/)')
    lib.np_decoder_new.argtypes = [p, p, p, z, i, i, i, p, z, p, z]
    lib.np_decoder_new.restype = p
    lib.np_decoder_free.argtypes = [p]
    lib.np_decoder_free.restype = None
    lib.np_decoder_set_tools.argtypes = [p, ctypes.c_char_p, ctypes.c_char_p, p, z]
    lib.np_decoder_set_tools.restype = i
    lib.np_decoder_start.argtypes = [p, ctypes.c_char_p, z, i, ctypes.c_float, p, z]
    lib.np_decoder_start.restype = i
    lib.np_decoder_step.argtypes = [p, p, z, p, z]
    lib.np_decoder_step.restype = i
    lib.np_decoder_state.argtypes = [p]
    lib.np_decoder_state.restype = i
    lib.np_decoder_result.argtypes = [p, p, z]
    lib.np_decoder_result.restype = ctypes.c_int64
    _LIBRARIES[path] = lib
    return lib


_LIBRARIES = {}


class Engine:
    """One model in memory with one cached request (KV cache). Not thread-safe across requests."""

    def __init__(self, model, *, library=None, threads=0, window=0, decode_skip=None):
        self.model = Path(model).resolve(strict=True)
        self.library = find_library(library)
        if os.name == 'nt':
            self._dll_directory = os.add_dll_directory(str(self.library.parent))
        lib = self._lib = load_library(self.library)
        p, i, z = ctypes.c_void_p, ctypes.c_int32, ctypes.c_size_t
        lib.np_load.argtypes = [ctypes.c_char_p, ctypes.c_uint64, ctypes.c_uint64, i, p, z]
        lib.np_load.restype = p
        lib.np_eval.argtypes = [p, p, z, p, z, p, z]
        lib.np_eval.restype = i
        lib.np_reset.argtypes = [p]
        lib.np_reset.restype = None
        lib.np_free.argtypes = [p]
        lib.np_free.restype = None
        lib.np_confidence.argtypes = [p, ctypes.POINTER(ctypes.c_float), p, z]
        lib.np_confidence.restype = i
        lib.np_set_window.argtypes = [p, z, p, z]
        lib.np_set_window.restype = i
        lib.np_set_prefix.argtypes = [p, z, p, z]
        lib.np_set_prefix.restype = i
        lib.np_prefix_reused.argtypes = [p]
        lib.np_prefix_reused.restype = z
        lib.np_set_decode_skip.argtypes = [p, ctypes.c_uint64, p, z]
        lib.np_set_decode_skip.restype = i
        self._lock = threading.RLock()
        error = ctypes.create_string_buffer(2048)
        self._handle = lib.np_load(os.fsencode(self.model), 0, 0, threads, error, len(error))
        if not self._handle:
            raise RuntimeError(error.value.decode('utf-8', 'replace'))
        self._finalizer = weakref.finalize(self, lib.np_free, self._handle)
        with self.model.open('rb') as handle:
            self.geometry = struct.unpack('<48If', handle.read(196))
        self.vocab = self.geometry[6] or self.geometry[5]
        self.context = self.geometry[13]
        trained = self.geometry[45] | self.geometry[46] << 32
        # The layers this model was trained to skip at decode (header fields 45-46; empty for most models).
        self.skip_layers = tuple(i for i in range(self.geometry[10]) if trained >> i & 1)
        if window:
            self.set_window(window)
        if decode_skip:
            self.set_decode_skip(decode_skip)

    def _check(self, status, error):
        if status < 0:
            raise RuntimeError(error.value.decode('utf-8', 'replace'))
        return status

    def eval(self, tokens):
        """Feed tokens after the cached ones; logits for the next token."""
        tokens = np.ascontiguousarray(tokens, dtype=np.int32)
        if tokens.ndim != 1 or not tokens.size:
            raise ValueError('Tokens must be a nonempty vector')
        with self._lock:
            output = np.empty(self.vocab, np.float32)
            error = ctypes.create_string_buffer(2048)
            self._check(self._lib.np_eval(self._handle, tokens.ctypes.data, tokens.size,
                                          output.ctypes.data, output.size, error, len(error)), error)
            return output

    def set_window(self, window):
        """Attention window in tokens (0: the model's own). Drops the cached request."""
        with self._lock:
            error = ctypes.create_string_buffer(2048)
            self._check(self._lib.np_set_window(self._handle, window, error, len(error)), error)

    def set_prefix(self, count):
        """Keep the state after the first `count` prompt tokens. A later prompt that starts with the
        same tokens skips their prefill; results do not depend on whether the cache was hit."""
        with self._lock:
            error = ctypes.create_string_buffer(2048)
            self._check(self._lib.np_set_prefix(self._handle, count, error, len(error)), error)

    def set_decode_skip(self, layers):
        """Decode skip: generated tokens skip these layers; prompt tokens always run every layer.
        Off by default. `True` uses the model's own list (`skip_layers`), a list or '5-8,10' names the
        layers, and None/False/() turns it off. Lists the model was not trained with cost accuracy."""
        if layers is True:
            layers = self.skip_layers
        elif isinstance(layers, str):
            layers = parse_layers(layers)
        mask = sum(1 << int(layer) for layer in set(layers or ()))
        with self._lock:
            error = ctypes.create_string_buffer(2048)
            self._check(self._lib.np_set_decode_skip(self._handle, mask, error, len(error)), error)

    def prefix_reused(self):
        """Prompt tokens the last prefill took from the prefix cache."""
        with self._lock:
            return self._lib.np_prefix_reused(self._handle)

    def confidence(self):
        """The archive's confidence head score for the cached tokens, or None without a head."""
        with self._lock:
            value, error = ctypes.c_float(), ctypes.create_string_buffer(2048)
            present = self._check(self._lib.np_confidence(self._handle, ctypes.byref(value), error, len(error)), error)
            return value.value if present else None

    def reset(self):
        with self._lock:
            self._lib.np_reset(self._handle)

    def close(self):
        with self._lock:
            if self._handle:
                self._finalizer()
                self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class Decoder:
    """Token choice for tool calling, in the native library (native/grammar.cpp): only tokens that keep
    the answer a valid list of tool calls, plus the copy guard and enum debiasing. Python and the browser
    demo share this code. Enum debiasing learns across requests, so keep one decoder per model."""

    REQUIRE_CALL, REASONING, DEBIAS = 1, 2, 4
    DONE = 3

    def __init__(self, library, pieces, special, open_id, close_id, think_id, stop_ids):
        self._lib = load_library(find_library(library))
        blob = b''.join(pieces)
        offsets = np.zeros(len(pieces) + 1, np.uint32)
        offsets[1:] = np.cumsum([len(piece) for piece in pieces])
        flags = np.asarray(special, np.uint8)
        stops = np.asarray(sorted(stop_ids), np.int32)
        error = ctypes.create_string_buffer(2048)
        self._handle = self._lib.np_decoder_new(blob, offsets.ctypes.data, flags.ctypes.data, len(pieces), open_id,
                                                close_id, -1 if think_id is None else think_id, stops.ctypes.data,
                                                len(stops), error, len(error))
        if not self._handle:
            raise RuntimeError(error.value.decode('utf-8', 'replace'))
        self._finalizer = weakref.finalize(self, self._lib.np_decoder_free, self._handle)

    def _check(self, status, error):
        if status < 0:
            raise RuntimeError(error.value.decode('utf-8', 'replace'))
        return status

    def set_tools(self, tools, label_tokens):
        """`tools`: [{name, parameters}] with local $refs inlined; `label_tokens`: {enum label: first token}."""
        error = ctypes.create_string_buffer(2048)
        self._check(self._lib.np_decoder_set_tools(self._handle, json.dumps(tools).encode(),
                                                   json.dumps(label_tokens).encode(), error, len(error)), error)

    def start(self, query, *, require_call=False, reasoning=True, debias=True, copy_margin=4.0):
        data = query.encode('utf-8')
        flags = self.REQUIRE_CALL * bool(require_call) | self.REASONING * bool(reasoning) | self.DEBIAS * bool(debias)
        error = ctypes.create_string_buffer(2048)
        self._check(self._lib.np_decoder_start(self._handle, data, len(data), flags, copy_margin, error, len(error)), error)

    def step(self, logits):
        """Pick, record and take the next token for these logits; returns it."""
        logits = np.ascontiguousarray(logits, dtype=np.float32)
        error = ctypes.create_string_buffer(2048)
        return self._check(self._lib.np_decoder_step(self._handle, logits.ctypes.data, logits.size, error, len(error)), error)

    @property
    def done(self):
        return self._lib.np_decoder_state(self._handle) == self.DONE

    def result(self):
        """{"complete": bool, "calls": [...], "probabilities": [...]}"""
        size = self._lib.np_decoder_result(self._handle, None, 0)
        buffer = ctypes.create_string_buffer(size + 1)
        self._lib.np_decoder_result(self._handle, buffer, len(buffer))
        return json.loads(buffer.value.decode('utf-8'))

    def close(self):
        if self._handle:
            self._finalizer()
            self._handle = None
