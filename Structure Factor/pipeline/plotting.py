"""Diagnostic plots for the dual-engine S(q) pipeline.

Notation: q = reciprocal-space wavevector magnitude (DPD units).
The same plots are produced whichever engine (cic or exact) computed the
spectrum; the engine is named in the figure titles.

Plots produced per simulation:
  1. Time series of window-selection metrics (q_star, phi_rms, peak_height, PotEng)
     with the accepted window shaded and fallback windows explicitly labelled.
  2. Averaged partial S(q): S_AA, S_BB, S_AB, S_tot.
  3. Composition-contrast S_comp(q) = S_AA+S_BB−2·S_AB with q_star marked.
     If q_peak_search_max was restricted below q_compute_max, the cutoff is
     shown as a vertical line so the restriction is visible in the figure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend (safe for HPC and Agg-only environments)
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False


def _require_mpl():
    if not _MPL:
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )


def plot_window_selection(
    timesteps: List[int],
    q_star_fft: List[float],
    phi_rms: List[float],
    window_result,
    block_diagnostics,
    outpath: str | Path,
    sim_id: str = "",
    thermo_df=None,
    peak_height_fft: Optional[List[float]] = None,
    sq_window: Optional[tuple] = None,
    spectral_stationary: Optional[bool] = None,
) -> None:
    """Time-series plot of window-detection metrics with accepted window shaded.

    Panels (from top):
      1. FFT q_star per frame
      2. phi_rms per frame
      3. FFT peak height per frame (if provided)
      4. PotEng from thermo (if available)

    The detected stationary window is shaded in blue. If a fallback window was
    used, "FALLBACK WINDOW — not automatically equilibrated" is annotated in
    red. When *sq_window* is given as (start_step, end_step) — a pinned
    window from sq_window_steps or explicit bounds — it is outlined in orange
    so the frames the engine averaged can be compared against the detected
    window.
    """
    _require_mpl()

    ts = np.array(timesteps, dtype=float)
    qs = np.array(q_star_fft, dtype=float)
    phi = np.array(phi_rms, dtype=float)
    has_ph = peak_height_fft is not None and len(peak_height_fft) == len(timesteps)
    thermo_plot = None
    if ts.size:
        traj_lo = float(np.nanmin(ts))
        traj_hi = float(np.nanmax(ts))
    else:
        traj_lo = traj_hi = float("nan")
    has_thermo = (
        thermo_df is not None
        and not thermo_df.empty
        and "PotEng" in thermo_df.columns
        and "Step" in thermo_df.columns
    )
    if has_thermo:
        # Thermo logs often include ramps/annealing before the production
        # trajectory begins. Plot only the overlapping interval; otherwise
        # sharex=True compresses all structural points into the far right of
        # the figure and an initialization-energy spike flattens PotEng.
        thermo_plot = thermo_df[
            (thermo_df["Step"] >= traj_lo) & (thermo_df["Step"] <= traj_hi)
        ]
        has_thermo = not thermo_plot.empty
    nrows = 2 + int(has_ph) + int(has_thermo)

    fig, axes = plt.subplots(nrows, 1, figsize=(9, 2.8 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    def shade(ax):
        if window_result is not None:
            ax.axvspan(
                window_result.window_start_step,
                window_result.window_end_step,
                alpha=0.15, color="steelblue", label="Candidate stationary window",
            )
        if sq_window is not None:
            ax.axvspan(
                sq_window[0], sq_window[1],
                facecolor="none", edgecolor="darkorange", lw=1.2, hatch="///",
                alpha=0.9, label="Sq averaging window (pinned)",
            )

    iax = 0

    # Panel: q_star
    axes[iax].plot(ts, qs, ".", ms=4, color="C0", label=r"$q^*$ (FFT diagnostic)")
    shade(axes[iax])
    axes[iax].set_ylabel(r"$q^*_\mathrm{FFT}$ (DPD units)")
    axes[iax].legend(fontsize=7, loc="upper left")
    if sim_id:
        axes[iax].set_title(f"{sim_id} — window-selection metrics", fontsize=9)
    iax += 1

    # Panel: phi_rms
    axes[iax].plot(ts, phi, ".", ms=4, color="C1", label=r"$\phi_\mathrm{rms}$")
    shade(axes[iax])
    axes[iax].set_ylabel(r"$\phi_\mathrm{rms}$")
    axes[iax].legend(fontsize=7, loc="upper left")
    iax += 1

    # Panel: FFT peak height (if provided)
    if has_ph:
        ph = np.array(peak_height_fft, dtype=float)
        axes[iax].plot(ts, ph, ".", ms=4, color="C3", label="FFT peak height")
        shade(axes[iax])
        axes[iax].set_ylabel("FFT peak height")
        axes[iax].legend(fontsize=7, loc="upper left")
        iax += 1

    # Panel: PotEng (if thermo available)
    if has_thermo:
        thstep = thermo_plot["Step"].to_numpy(float)
        pe = thermo_plot["PotEng"].to_numpy(float)
        axes[iax].plot(thstep, pe, lw=0.8, color="C2", label="PotEng")
        shade(axes[iax])
        axes[iax].set_ylabel("PotEng")
        axes[iax].legend(fontsize=7, loc="upper left")
        iax += 1

    axes[-1].set_xlabel("Timestep")
    if np.isfinite(traj_lo) and np.isfinite(traj_hi) and traj_hi > traj_lo:
        axes[-1].set_xlim(traj_lo, traj_hi)

    # Fallback annotation on every panel
    if window_result is not None and window_result.fallback_used:
        for ax in axes:
            ax.annotate(
                "FALLBACK WINDOW\n(not automatically equilibrated)",
                xy=(0.02, 0.96), xycoords="axes fraction",
                color="red", fontsize=7, va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="red"),
            )
    if spectral_stationary is False:
        for ax in axes:
            ax.annotate(
                "S(q) PLATEAU CHECK FAILED",
                xy=(0.98, 0.96), xycoords="axes fraction",
                color="darkred", fontsize=7, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="darkred"),
            )

    fig.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sq_partials(
    q: np.ndarray,
    S_AA: np.ndarray,
    S_BB: np.ndarray,
    S_AB: np.ndarray,
    S_tot: np.ndarray,
    outpath: str | Path,
    sim_id: str = "",
    q_plot_min: float = 0.2,
    q_plot_max: float = 4.0,
    sq_method: str = "",
) -> None:
    """Plot partial and total S(q) curves from either engine."""
    _require_mpl()
    fig, ax = plt.subplots(figsize=(7, 4))
    mask = (q >= q_plot_min) & (q <= q_plot_max) & np.isfinite(q)
    ax.plot(q[mask], S_AA[mask], label=r"$S_{AA}(q)$", lw=1.5, color="C0")
    ax.plot(q[mask], S_BB[mask], label=r"$S_{BB}(q)$", lw=1.5, color="C1")
    ax.plot(q[mask], S_AB[mask], label=r"$S_{AB}(q)$", lw=1.5, color="C2", ls="--")
    ax.plot(q[mask], S_tot[mask], label=r"$S_\mathrm{tot}(q)$", lw=1.0, color="k", ls=":")
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set_xlabel(r"$q$ (DPD units)")
    ax.set_ylabel(r"$S(q)$ / $N^{-1}$")
    engine_note = f" [{sq_method}]" if sq_method else ""
    title = (
        f"{sim_id} — partial $S(q)${engine_note}"
        if sim_id else f"Partial $S(q)${engine_note}"
    )
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)
    ax.set_xlim(q_plot_min, q_plot_max)
    fig.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_composition_sq(
    q: np.ndarray,
    S_comp: np.ndarray,
    q_star: float,
    L0: float,
    outpath: str | Path,
    sim_id: str = "",
    q_plot_min: float = 0.2,
    q_plot_max: float = 4.0,
    q_peak_search_max: Optional[float] = None,
    q_fft: Optional[np.ndarray] = None,
    Sq_fft: Optional[np.ndarray] = None,
    peak_detection_method: str = "",
    n_peak_candidates: int = 0,
    sq_method: str = "",
) -> None:
    """Plot S_comp(q) with q_star marked; optionally overlay FFT diagnostic.

    If q_peak_search_max was restricted below q_plot_max, a vertical dashed
    line at q_peak_search_max shows the upper bound of the peak search.
    """
    _require_mpl()
    fig, ax = plt.subplots(figsize=(7, 4))
    mask = (
        (q >= q_plot_min) & (q <= q_plot_max)
        & np.isfinite(q) & np.isfinite(S_comp) & (S_comp > 0)
    )
    ax.plot(
        q[mask], S_comp[mask],
        label=r"$S_\mathrm{comp}(q) = S_{AA}+S_{BB}-2S_{AB}$", lw=2.0, color="C3",
    )

    # FFT diagnostic overlay (rescaled for visual comparison only)
    if q_fft is not None and Sq_fft is not None:
        mask_fft = (
            (q_fft >= q_plot_min) & (q_fft <= q_plot_max)
            & np.isfinite(q_fft) & np.isfinite(Sq_fft) & (Sq_fft > 0)
        )
        if np.any(mask_fft):
            Sq_fft_sub = Sq_fft[mask_fft]
            S_comp_sub = S_comp[mask]
            if S_comp_sub.size:
                scale = np.nanmax(S_comp_sub) / max(float(np.nanmax(Sq_fft_sub)), 1e-12)
                ax.plot(
                    q_fft[mask_fft], Sq_fft_sub * scale,
                    label=r"$S(q)$ FFT [rescaled, diagnostic only]",
                    lw=1.2, color="C4", ls="--", alpha=0.8,
                )

    ax.set_yscale("log")

    # q_star marker
    if np.isfinite(q_star) and q_star > 0:
        ax.axvline(q_star, color="red", lw=1.2, ls="--",
                   label=rf"$q^* = {q_star:.3f}$, $L_0 = {L0:.2f}$")
        ax.text(
            0.02, 0.06,
            f"$L_0 = {L0:.2f}$\n$q^* = {q_star:.3f}$",
            transform=ax.transAxes, color="red", fontsize=7, va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.8),
        )

    # Peak-search upper bound annotation (only when explicitly restricted)
    if q_peak_search_max is not None and q_peak_search_max < q_plot_max:
        ax.axvline(
            q_peak_search_max, color="gray", lw=1.0, ls=":",
            label=rf"$q_\mathrm{{peak,max}} = {q_peak_search_max:.3f}$ (search limit)",
        )

    ax.set_xlabel(r"$q$ (DPD units)")
    ax.set_ylabel(r"$S_\mathrm{comp}(q)$ / $N^{-1}$")
    engine_note = f" [{sq_method}]" if sq_method else ""
    method_note = f" [{peak_detection_method}, {n_peak_candidates} candidates]" \
        if peak_detection_method else ""
    title = (
        f"{sim_id} — log-y composition-contrast $S(q)${engine_note}{method_note}"
        if sim_id else f"Log-y composition-contrast $S(q)${engine_note}{method_note}"
    )
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=7)
    ax.set_xlim(q_plot_min, q_plot_max)
    ax.grid(True, which="both", alpha=0.18)
    fig.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_log_sq_screening(
    q: np.ndarray,
    S_comp: np.ndarray,
    S_cc_normalized: np.ndarray,
    outpath: str | Path,
    *,
    q_star: float = np.nan,
    L0: float = np.nan,
    peak_detected: bool = False,
    sim_id: str = "",
    sq_method: str = "",
    q_plot_min: Optional[float] = None,
    q_plot_max: Optional[float] = None,
) -> None:
    """Write a compact log-y S(q) plot intended for rapid campaign screening."""
    _require_mpl()
    q = np.asarray(q, dtype=float)
    raw = np.asarray(S_comp, dtype=float)
    normalized = np.asarray(S_cc_normalized, dtype=float)
    mask = np.isfinite(q)
    if q_plot_min is not None:
        mask &= q >= q_plot_min
    if q_plot_max is not None:
        mask &= q <= q_plot_max

    fig, ax = plt.subplots(figsize=(8, 4.8))
    raw_mask = mask & np.isfinite(raw) & (raw > 0)
    norm_mask = mask & np.isfinite(normalized) & (normalized > 0)
    ax.plot(
        q[raw_mask], raw[raw_mask], lw=1.5, color="C3",
        label="raw composition contrast",
    )
    ax.plot(
        q[norm_mask], normalized[norm_mask], lw=1.5, color="C0",
        label="composition-normalized concentration",
    )
    ax.set_yscale("log")
    if peak_detected and np.isfinite(q_star) and q_star > 0:
        ax.axvline(
            q_star, color="black", ls="--", lw=1.0,
            label=rf"significant peak: $q^*={q_star:.3f}$, $L_0={L0:.2f}$",
        )
        status = "significant finite-q peak detected"
    else:
        status = "no significant finite-q peak detected"
        ax.text(
            0.02, 0.04, status, transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.5", alpha=0.9),
        )
    engine_note = f" [{sq_method}]" if sq_method else ""
    ax.set_title(f"{sim_id} — log S(q){engine_note}\n{status}", fontsize=9)
    ax.set_xlabel(r"$q$ (DPD units)")
    ax.set_ylabel(r"$S(q)$")
    ax.grid(True, which="both", alpha=0.18)
    ax.legend(fontsize=8)
    fig.tight_layout()
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_all_plots(
    sim_id: str,
    plot_dir: Path,
    timesteps: List[int],
    q_star_fft: List[float],
    phi_rms: List[float],
    window_result,
    sq_result,
    cfg: Dict[str, Any],
    thermo_df=None,
    q_peak_search_max: Optional[float] = None,
    q_fft_avg: Optional[np.ndarray] = None,
    Sq_fft_avg: Optional[np.ndarray] = None,
    peak_height_fft: Optional[List[float]] = None,
    resolved: Optional[Dict[str, Any]] = None,
    sq_window: Optional[tuple] = None,
    sq_method: str = "",
    spectral_stationary: Optional[bool] = None,
) -> None:
    """Create all diagnostic plots for one simulation (either engine).

    ``resolved`` is the per-simulation resolved q-grid dict from
    resolve_q_grid; config values may be the string 'auto' and are not usable
    as numeric plot bounds directly. ``sq_method`` names the engine in titles.
    """
    if not _MPL:
        return

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    if resolved is not None:
        q_lo = float(resolved["q_peak_search_min"])
        q_hi = float(resolved["q_compute_max"])
    elif sq_result is not None:
        # Fall back to the data range itself (finite bins only)
        q_arr = np.asarray(sq_result.q, dtype=float)
        q_lo = float(np.nanmin(q_arr))
        q_hi = float(np.nanmax(q_arr))
    else:
        q_lo, q_hi = 0.2, float(cfg.get("q_compute_max", 4.0))

    plot_window_selection(
        timesteps, q_star_fft, phi_rms,
        window_result,
        window_result.block_diagnostics if window_result else None,
        plot_dir / f"{sim_id}_window_selection.png",
        sim_id=sim_id,
        thermo_df=thermo_df,
        peak_height_fft=peak_height_fft,
        sq_window=sq_window,
        spectral_stationary=spectral_stationary,
    )

    if sq_result is not None:
        plot_sq_partials(
            sq_result.q, sq_result.S_AA, sq_result.S_BB,
            sq_result.S_AB, sq_result.S_tot,
            plot_dir / f"{sim_id}_sq_partials.png",
            sim_id=sim_id, q_plot_min=q_lo, q_plot_max=q_hi,
            sq_method=sq_method,
        )
        plot_composition_sq(
            sq_result.q, sq_result.S_comp,
            sq_result.q_star, sq_result.L0,
            plot_dir / f"{sim_id}_composition_sq.png",
            sim_id=sim_id,
            q_plot_min=q_lo,
            q_plot_max=q_hi,
            q_peak_search_max=q_peak_search_max,
            q_fft=q_fft_avg,
            Sq_fft=Sq_fft_avg,
            peak_detection_method=sq_result.peak_detection_method,
            n_peak_candidates=sq_result.n_peak_candidates,
            sq_method=sq_method,
        )
