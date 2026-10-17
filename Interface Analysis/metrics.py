"""Aggregation of per-frame metrics into simulation summaries.

Frames from one trajectory are usually CORRELATED (consecutive configurations
share structure), so naive standard errors over frames are optimistic. Every
summary carries n_frames and, when enabled, an integrated-autocorrelation
estimate tau_int and effective sample size N_eff = N / (2 tau_int) for key
series — labeled experimental because the series here are short (10-200
frames), where windowed autocorrelation estimates are noisy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def aggregate_series(values: Sequence[float]) -> Dict[str, float]:
    """mean/std/median/IQR/min/max/n over the finite entries of a series."""
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        nan = float("nan")
        return {"mean": nan, "std": nan, "median": nan, "iqr": nan,
                "min": nan, "max": nan, "n": 0}
    q1, q3 = np.percentile(arr, [25, 75])
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "iqr": float(q3 - q1),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


def integrated_autocorr(values: Sequence[float]) -> Dict[str, float]:
    """Integrated autocorrelation time via Sokal windowing (c = 5).

    Returns tau_int (in frame intervals) and N_eff = N / (2 tau_int).
    For short or constant series returns tau = nan with a reason.
    EXPERIMENTAL for the short series typical here — treat as indicative.
    """
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    out = {"tau_int": float("nan"), "n_eff": float("nan"), "reliable": False}
    if n < 8:
        out["note"] = "series too short for autocorrelation estimate"
        return out
    x = arr - arr.mean()
    var = float(np.dot(x, x) / n)
    if var <= 0:
        out["note"] = "zero-variance series"
        return out
    # FFT autocorrelation
    m = 1 << (2 * n - 1).bit_length()
    F = np.fft.rfft(x, m)
    ac = np.fft.irfft(F * np.conj(F))[:n].real / (var * n)
    tau = 0.5
    for t in range(1, n):
        tau += ac[t]
        if t >= 5.0 * tau:          # Sokal window c = 5
            break
    tau = max(float(tau), 0.5)
    out["tau_int"] = tau
    out["n_eff"] = n / (2.0 * tau)
    out["reliable"] = bool(n >= 20 * tau)
    if not out["reliable"]:
        out["note"] = "series shorter than ~20 tau; tau_int is a rough lower bound"
    return out


def summarize_frame_rows(
    rows: List[Dict[str, Any]],
    autocorr_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Aggregate a list of flat per-frame metric dicts.

    Numeric keys get mean/std/median/iqr/min/max; selected keys additionally
    get tau_int / N_eff. Non-numeric keys are ignored.
    """
    if not rows:
        return {"n_frames": 0}
    summary: Dict[str, Any] = {"n_frames": len(rows)}
    keys = set()
    for r in rows:
        keys.update(k for k, v in r.items() if isinstance(v, (int, float)) and
                    not isinstance(v, bool))
    for key in sorted(keys):
        series = [r.get(key) for r in rows]
        agg = aggregate_series(series)
        for stat, val in agg.items():
            summary[f"{key}_{stat}"] = val
    for key in (autocorr_keys or []):
        series = [r.get(key) for r in rows]
        ac = integrated_autocorr(series)
        summary[f"{key}_tau_int"] = ac["tau_int"]
        summary[f"{key}_n_eff"] = ac["n_eff"]
        summary[f"{key}_tau_reliable"] = ac["reliable"]
    return summary
