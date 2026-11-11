"""ctypes binding to the native engine (native/, built as libspindle)."""
import ctypes
import os
import struct
import sys
import threading
import weakref
from pathlib import Path

import numpy as np

ABI_VERSION = 1
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


class Engine:
    """One model in memory with one cached request (KV cache). Not thread-safe across requests."""

    def __init__(self, model, *, library=None, threads=4, window=0):
        self.model = Path(model).resolve(strict=True)
        self.library = find_library(library)
        if os.name == 'nt':
            self._dll_directory = os.add_dll_directory(str(self.library.parent))
        lib = self._lib = ctypes.CDLL(str(self.library))
        p, i, z = ctypes.c_void_p, ctypes.c_int32, ctypes.c_size_t
        lib.np_abi_version.restype = ctypes.c_uint32
        if lib.np_abi_version() != ABI_VERSION:
            raise RuntimeError('The native engine and this Python package are out of date with each other')
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
        if window:
            self.set_window(window)

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
        """Keep the state after the first `count` prompt tokens (whole prefill chunks only).
        A later prompt that starts with the same tokens skips their prefill; results do not change."""
        with self._lock:
            error = ctypes.create_string_buffer(2048)
            self._check(self._lib.np_set_prefix(self._handle, count, error, len(error)), error)

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
