import numpy as np

from spindle import archive


def _codebook():
    """Gaussian quantiles: close to the Lloyd-Max codebooks real archives carry."""
    from statistics import NormalDist
    books = [[NormalDist().inv_cdf((i + 0.5) / (1 << bits)) for i in range(1 << bits)] for bits in (2, 3, 4)]
    return (np.concatenate(books) / np.sqrt(archive.GROUP)).astype(np.float32)


def test_cq4_pack_unpack():
    codebook = _codebook()
    w = np.random.RandomState(0).randn(8, 200).astype(np.float32)
    packed = archive.pack_cq4(w, codebook)
    assert len(packed) == 8 * 256 // 2 + 8 * 2 * 2  # codes + FP16 norms (two groups per row)
    restored = archive.unpack_cq(packed, w.shape, 4, codebook)
    assert restored.shape == w.shape
    assert np.corrcoef(w.ravel(), restored.ravel())[0, 1] > 0.98


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
