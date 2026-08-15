"""Optional GPU-accelerated rank computation via cupy (user-granted dep).

The dataset-analysis pipeline's dominant cost is per-era ``rankdata``
(sorting). When cupy is importable, ranking runs on the GPU; the ranks
themselves are exact values computed with a stable sort + tie-averaging, so
the results are **bit-identical** to ``scipy.stats.rankdata(method="average")``
and the rest of the pipeline stays on CPU. When cupy is unavailable (or a GPU
call fails), this module degrades to scipy — same outputs, no crash, with a
logged warning on runtime GPU failure.

Deliberately NOT part of ``nmr._transforms``: that module is embedded by
value into deployed ``predict.pkl`` artifacts, which must stay
numpy/scipy-only for the hosted runtime. Loading is lazy — importing this
module never imports cupy.
"""

from __future__ import annotations

import glob
import logging
import os
import site

import numpy as np
import scipy.stats

logger = logging.getLogger("nmr._gpu")

_CUPY = None
_CUPY_LOADED = False


def _bootstrap_cuda_path() -> None:
    """Put NVIDIA runtime DLL dirs (from the nvidia-* wheels) on PATH.

    The cupy-cuda12x wheel does not bundle the CUDA runtime on Windows; the
    nvidia-cublas-cu12 / nvidia-cuda-runtime-cu12 / ... wheels ship the DLLs,
    which only resolve when their ``bin`` directories are on PATH.
    """
    if os.environ.get("CUDA_PATH"):
        return
    try:
        site_packages = site.getsitepackages()[0]
    except (AttributeError, IndexError):
        site_packages = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bins = sorted(
        os.path.dirname(path)
        for path in glob.glob(os.path.join(site_packages, "nvidia", "*", "bin"))
    )
    if bins:
        os.environ["PATH"] = os.pathsep.join(bins + [os.environ.get("PATH", "")])
        # point CUDA_PATH at a package root whose bin/ holds the runtime DLLs
        # (silences cupy's environment probe; the DLLs resolve via PATH)
        os.environ["CUDA_PATH"] = os.path.dirname(bins[0])


def _load_cupy():
    """Import cupy once (lazy); None when unavailable."""
    global _CUPY, _CUPY_LOADED
    if _CUPY_LOADED:
        return _CUPY
    _CUPY_LOADED = True
    _bootstrap_cuda_path()
    try:
        import cupy  # noqa: PLC0415 — lazy import keeps `import nmr` cheap

        _CUPY = cupy
    except Exception:  # pragma: no cover — environment-dependent
        _CUPY = None
    return _CUPY


def _rankdata_gpu_1d(values: np.ndarray) -> np.ndarray:
    """Exact average-rank of a 1-D array on the GPU (NaN -> NaN)."""
    cp = _CUPY
    x = cp.asarray(values, dtype=cp.float64)
    finite = cp.isfinite(x)
    ranks = cp.full(values.size, cp.nan)
    if not bool(cp.any(finite)):
        return cp.asnumpy(ranks)
    xf = x[finite]
    n = xf.size
    order = cp.argsort(xf, kind="stable")
    sorted_vals = xf[order]
    changes = cp.nonzero(sorted_vals[1:] != sorted_vals[:-1])[0] + 1
    starts = cp.concatenate([cp.zeros(1, dtype=cp.int64), changes])
    ends = cp.concatenate([changes, cp.array([n], dtype=cp.int64)])
    counts = ends - starts
    dense = cp.arange(1, n + 1, dtype=cp.float64)  # 1-based, like scipy
    group_mean = cp.add.reduceat(dense, starts) / counts
    group_idx = cp.searchsorted(starts, cp.arange(n, dtype=cp.int64), side="right") - 1
    ranks_sorted = group_mean[group_idx]
    ranks_finite = ranks_sorted[cp.argsort(order)]
    ranks[finite] = ranks_finite
    return cp.asnumpy(ranks)


def _rankdata_gpu_matrix(values: np.ndarray, axis: int) -> np.ndarray:
    """Exact average-rank of a 2-D array along ``axis`` on the GPU.

    Port of scipy's own ``_rankdata`` algorithm (argsort -> group-start
    flags -> group mid-ranks -> put_along_axis), which is array-API
    compatible by design, with cupy ops. Non-finite entries rank as NaN
    (scipy 1.17's all-NaN 'propagate' poisoning is not replicated).
    """
    cp = _CUPY
    x = cp.asarray(values, dtype=cp.float64)
    finite = cp.isfinite(x)
    x = cp.where(finite, x, cp.inf)
    swap = axis == 0
    if swap:
        x = x.T
        finite = finite.T
    shape = x.shape
    n = shape[-1]
    j = cp.argsort(x, axis=-1, kind="stable")
    ordinal = cp.broadcast_to(cp.arange(1, n + 1, dtype=cp.float64), shape)
    y = cp.take_along_axis(x, j, axis=-1)
    i = cp.concatenate(
        [cp.ones(shape[:-1] + (1,), dtype=cp.bool_), y[..., :-1] != y[..., 1:]],
        axis=-1,
    )
    flat = cp.arange(y.size, dtype=cp.int64)[i.reshape(-1)]
    counts = cp.diff(flat)
    counts = cp.concatenate(
        [counts, cp.array([y.size - int(flat[-1])], dtype=cp.int64)]
    )
    ranks = ordinal[i] + (cp.asarray(counts, dtype=cp.float64) - 1) / 2.0
    ranks = cp.reshape(cp.repeat(ranks, counts), shape)
    out = cp.empty_like(ranks)
    cp.put_along_axis(out, j, ranks, axis=-1)
    if swap:
        out = out.T
        finite = finite.T
    out = cp.where(finite, out, cp.nan)
    return cp.asnumpy(out)


def _rankdata_scipy_isolated(array: np.ndarray, axis: int | None) -> np.ndarray:
    """scipy average-rank with NaN isolated at NaN positions (no poisoning).

    Matches the GPU path's NaN semantics: finite values are ranked exactly as
    scipy would rank them, and NaN slots stay NaN. On finite input this is
    bit-identical to ``scipy.stats.rankdata(method="average")``.
    """
    if axis is None:
        flat = array.reshape(-1)
        out = np.full(flat.size, np.nan, dtype=np.float64)
        finite = np.isfinite(flat)
        if finite.any():
            out[finite] = scipy.stats.rankdata(flat[finite], method="average")
        return out.reshape(array.shape)
    ax = axis % array.ndim
    moved = np.moveaxis(array, ax, -1)
    out = np.full(moved.shape, np.nan, dtype=np.float64)
    for idx in np.ndindex(moved.shape[:-1]):
        sl = moved[idx]
        finite = np.isfinite(sl)
        if finite.any():
            out[idx][finite] = scipy.stats.rankdata(sl[finite], method="average")
    return np.moveaxis(out, -1, ax)


def rankdata(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Average-rank matching scipy.stats.rankdata(method="average") exactly.

    GPU when cupy is available, scipy otherwise; outputs are bit-identical
    on finite data. With NaN present, scipy 1.17 poisons the whole output
    while this function isolates NaN at NaN positions on **both** paths —
    intentionally more correct, and uniform across machines. ``axis``
    mirrors scipy's semantics: None ranks the flattened array.
    """
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array.copy()
    cp = _load_cupy()
    if cp is None:
        return _rankdata_scipy_isolated(array, axis=axis)
    try:
        if axis is None:
            return _rankdata_gpu_1d(array.reshape(-1))
        if axis in (0, -2, 1, -1):
            return _rankdata_gpu_matrix(array, axis=axis)
        raise ValueError(f"unsupported axis {axis!r}")
    except Exception as exc:  # GPU runtime failure (OOM, driver): CPU fallback
        logger.warning("[rankdata] GPU path failed (%s); falling back to scipy", exc)
        return _rankdata_scipy_isolated(array, axis=axis)
