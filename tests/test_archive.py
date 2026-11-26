import pytest
import numpy as np

from spindle import archive, lattice


def _codebook():
    """Gaussian quantiles: close to the Lloyd-Max codebooks real archives carry."""
    from statistics import NormalDist
    books = [[NormalDist().inv_cdf((i + 0.5) / (1 << bits)) for i in range(1 << bits)] for bits in (2, 3, 4)]
    return (np.concatenate(books) / np.sqrt(archive.GROUP)).astype(np.float32)


@pytest.mark.parametrize('bits, fit', [(4, 0.98), (2, 0.9), (lattice.E8, 0.94)])
def test_cq_pack_unpack(bits, fit):
    codebook = _codebook()
    w = np.random.RandomState(0).randn(8, 200).astype(np.float32)
    packed = archive.pack_cq(w, codebook, bits)
    code_bits = 2 if bits == lattice.E8 else bits
    assert len(packed) == 8 * 256 * code_bits // 8 + 8 * 2 * 2  # codes + FP16 norms (two groups per row)
    restored = archive.unpack_cq(packed, w.shape, bits, codebook)
    assert restored.shape == w.shape
    assert np.corrcoef(w.ravel(), restored.ravel())[0, 1] > fit


def test_e8_table_and_training_rounding():
    points, _ = lattice.table()
    assert len(points) == 56881 and len(np.unique(points, axis=0)) == len(points)
    w = np.random.RandomState(1).randn(16, 384).astype(np.float32)
    stored = archive.unpack_cq(archive.pack_cq(w, _codebook(), lattice.E8), w.shape, lattice.E8, _codebook())
    pytest.importorskip('jax')  # training extras only
    from spindle.model.quantize import e8_quantize
    np.testing.assert_allclose(np.asarray(e8_quantize(w)), stored, rtol=1e-3, atol=1e-5)


def test_runtime_lora_roundtrip(model_path):
    raw = model_path.read_bytes()
    lora = archive.read_lora(raw)
    if lora is None:
        lora = {name: {'A': np.zeros((20, 768, 4), np.float32), 'B': np.zeros((20, 4, 8), np.float32)}
                for name in archive.LORA_TARGETS}
    bare = archive.strip_lora(raw)
    assert archive.read_lora(bare) is None
    again = archive.write_lora(bare, lora)
    restored = archive.read_lora(again)
    for name in archive.LORA_TARGETS:
        np.testing.assert_allclose(restored[name]['A'], lora[name]['A'].astype(np.float16), rtol=0, atol=0)
