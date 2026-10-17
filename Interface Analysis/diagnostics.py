"""Diagnostic plots for one simulation (last analyzed frame + time series).

Every figure is written under <sim_out>/plots/ at publication-friendly DPI.
The figures are designed to make FAILURES obvious: crossings drawn on the
composition slab, defect maps with explicit legends, fit overlays on
profiles, and spectra with the damped low-q region visible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

from .frame_analysis import FrameAnalysis  # noqa: E402
from .interface_detection import COL_CLASS_NAMES  # noqa: E402
from .io_utils import LogFn  # noqa: E402
from .width import _tanh_model  # noqa: E402

_DPI = 150

_DEFECT_COLORS = ["#d9d9d9", "#e41a1c", "#377eb8", "#ff7f00", "#984ea3", "#4daf4a"]


def _save(fig, path: Path, log: Optional[LogFn]) -> None:
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    if log:
        log(f"plot: {path.name}")


def make_all_plots(sim_id: str, plot_dir: Path, det: FrameAnalysis,
                   frame_df: pd.DataFrame, log: Optional[LogFn] = None) -> None:
    """Generate the full diagnostic set; individual failures are logged, not fatal."""
    plot_dir = Path(plot_dir)
    plotters = [
        ("field_slices", _plot_field_slices),
        ("crossings_overlay", _plot_crossings_overlay),
        ("height_maps", _plot_height_maps),
        ("height_histograms", _plot_height_histograms),
        ("spacing", _plot_spacing),
        ("width", _plot_width),
        ("penetration", _plot_penetration),
        ("profiles", _plot_profiles),
        ("height_spectra", _plot_height_spectra),
        ("normal_angles", _plot_normal_angles),
        ("defect_map", _plot_defect_map),
        ("topology", _plot_topology),
    ]
    for name, fn in plotters:
        try:
            fn(sim_id, plot_dir, det, log)
        except Exception as exc:  # keep going; a missing plot must not kill the run
            if log:
                log(f"ERROR: plot '{name}' failed: {exc}")
    try:
        _plot_timeseries(sim_id, plot_dir, frame_df, log)
    except Exception as exc:
        if log:
            log(f"ERROR: plot 'timeseries' failed: {exc}")


# ---------------------------------------------------------------------------
def _plot_field_slices(sim_id, plot_dir, det: FrameAnalysis, log):
    comp, fields = det.comp, det.fields
    k = comp.phi.shape[2] // 2
    lo, hi = comp.grid.box_lo, comp.grid.box_hi
    ext = [lo[0], hi[0], lo[1], hi[1]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    panels = [(fields.rho_A, "rho_A", "Reds", None, None),
              (fields.rho_B, "rho_B", "Blues", None, None),
              (comp.phi, "phi", "RdBu_r", -1.0, 1.0)]
    for ax, (f, name, cmap, vmin, vmax) in zip(axes, panels):
        im = ax.imshow(f[:, :, k].T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=ext)
        ax.set_title(f"{name} (z mid-slice, step {det.timestep})")
        ax.set_xlabel("x [$r_c$]"); ax.set_ylabel("y [$r_c$]")
        fig.colorbar(im, ax=ax, shrink=0.85)
    fig.suptitle(sim_id)
    _save(fig, plot_dir / f"{sim_id}_field_slices.png", log)


def _plot_crossings_overlay(sim_id, plot_dir, det: FrameAnalysis, log):
    slab, iset = det.slab, det.iset
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), constrained_layout=True)
    for ax, iv in zip(axes, [slab.shape[1] // 4, (3 * slab.shape[1]) // 4]):
        im = ax.imshow(slab.phi_uvw[:, iv, :].T, origin="lower", cmap="RdBu_r",
                       vmin=-1, vmax=1, aspect="auto",
                       extent=[slab.u[0], slab.u[-1], slab.w[0], slab.w[-1]])
        for j in range(iset.n_interfaces):
            hh = iset.h[j, :, iv]
            m = np.isfinite(hh)
            ax.plot(slab.u[m], hh[m], ".", color="k", ms=2)
            ax.axhline(iset.plane_ideal[j], color="k", lw=0.4, ls=":", alpha=0.5)
        ax.set_xlabel("u [$r_c$]"); ax.set_ylabel("w along $n_0$ [$r_c$]")
        ax.set_title(f"phi(u, w) at v={slab.v[iv]:.1f} + crossings")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"{sim_id}: detected zero crossings (dotted lines = ideal planes)")
    _save(fig, plot_dir / f"{sim_id}_crossings_overlay.png", log)


def _grid_of_maps(maps, titles, suptitle, cmap, path, log, extent, sym=False,
                  cbar_label=""):
    n = len(maps)
    if n == 0:
        return
    ncol = min(4, n)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.3 * nrow),
                             constrained_layout=True, squeeze=False)
    finite_all = np.concatenate([m[np.isfinite(m)].ravel() for m in maps if np.isfinite(m).any()])
    if finite_all.size == 0:
        return
    if sym:
        vmax = float(np.percentile(np.abs(finite_all), 99))
        vmin = -vmax
    else:
        vmin, vmax = np.percentile(finite_all, [1, 99])
    im = None
    for i, (m, t) in enumerate(zip(maps, titles)):
        ax = axes[i // ncol][i % ncol]
        im = ax.imshow(m.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=extent, aspect="auto")
        ax.set_title(t, fontsize=9)
        ax.set_xlabel("u"); ax.set_ylabel("v")
    for i in range(n, nrow * ncol):
        axes[i // ncol][i % ncol].axis("off")
    if im is not None:
        fig.colorbar(im, ax=[a for row in axes for a in row], shrink=0.7,
                     label=cbar_label)
    fig.suptitle(suptitle)
    _save(fig, path, log)


def _plot_height_maps(sim_id, plot_dir, det: FrameAnalysis, log):
    iset, slab = det.iset, det.slab
    ext = [slab.u[0], slab.u[-1], slab.v[0], slab.v[-1]]
    maps, titles = [], []
    for j in range(iset.n_interfaces):
        dh = iset.h[j] - iset.plane_mean[j]
        maps.append(dh)
        titles.append(f"iface {j} ({'R' if iset.plane_type[j] > 0 else 'F'}), "
                      f"valid {iset.valid_fraction[j]:.2f}")
    _grid_of_maps(maps, titles, f"{sim_id}: height fluctuation maps "
                  r"$\delta h_j(u,v)$ [$r_c$] (step %d)" % det.timestep,
                  "viridis", plot_dir / f"{sim_id}_height_maps.png", log, ext,
                  sym=False, cbar_label=r"$\delta h$ [$r_c$]")


def _plot_height_histograms(sim_id, plot_dir, det: FrameAnalysis, log):
    iset = det.iset
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for j in range(iset.n_interfaces):
        vals = iset.h[j][np.isfinite(iset.h[j])] - iset.plane_mean[j]
        if vals.size > 8:
            ax.hist(vals, bins=40, histtype="step", density=True,
                    label=f"iface {j} (rms {np.std(vals):.2f})")
    ax.set_xlabel(r"$h_j - \overline{h_j}$ [$r_c$]")
    ax.set_ylabel("probability density")
    ax.set_title(f"{sim_id}: interface height distributions (step {det.timestep})")
    ax.legend(fontsize=7, ncols=2)
    _save(fig, plot_dir / f"{sim_id}_height_histograms.png", log)


def _plot_spacing(sim_id, plot_dir, det: FrameAnalysis, log):
    sp, slab = det.spacing, det.slab
    if not sp.local_maps:
        return
    ext = [slab.u[0], slab.u[-1], slab.v[0], slab.v[-1]]
    titles = [f"pair {p} ({d['type']})" for p, d in zip(sp.pair_indices, sp.per_pair)]
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    ax2.hist(sp.all_values, bins=50, density=True, alpha=0.8)
    ax2.axvline(sp.mean, color="k", label=f"mean {sp.mean:.2f}")
    if sp.L0_sq:
        ax2.axvline(sp.L0_sq, color="r", ls="--", label=f"S(q) $L_0$ {sp.L0_sq:.2f}")
    ax2.set_xlabel(r"local spacing $L_j(u,v)$ [$r_c$]")
    ax2.set_ylabel("probability density")
    ax2.set_title(f"{sim_id}: local lamellar spacing (CV {sp.cv:.3f})")
    ax2.legend()
    _save(fig2, plot_dir / f"{sim_id}_spacing_hist.png", log)
    _grid_of_maps(sp.local_maps, titles,
                  f"{sim_id}: local spacing maps $L_j(u,v)$ [$r_c$]",
                  "magma", plot_dir / f"{sim_id}_spacing_maps.png", log, ext,
                  cbar_label=r"$L_j$ [$r_c$]")


def _plot_width(sim_id, plot_dir, det: FrameAnalysis, log):
    wd, slab = det.width, det.slab
    ext = [slab.u[0], slab.u[-1], slab.v[0], slab.v[-1]]
    have = [(m, f"iface {j}") for j, m in enumerate(wd.width_maps)
            if np.isfinite(m).any()]
    if have:
        _grid_of_maps([m for m, _ in have], [t for _, t in have],
                      f"{sim_id}: local width maps (corrected $\\lambda$) [$r_c$]",
                      "cividis", plot_dir / f"{sim_id}_width_maps.png", log, ext,
                      cbar_label=r"$\lambda$ [$r_c$]")
    if wd.lam_values.size:
        fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
        ax.hist(wd.lam_values, bins=50, density=True, alpha=0.6, label="raw $\\lambda$")
        ax.hist(wd.lam_corrected, bins=50, density=True, histtype="step", lw=1.5,
                label="obliquity-corrected")
        ax.axvline(wd.mean_lam_corrected, color="k",
                   label=f"mean {wd.mean_lam_corrected:.3f}")
        ax.set_xlabel(r"tanh width $\lambda$ [$r_c$]")
        ax.set_ylabel("probability density")
        ax.set_title(f"{sim_id}: local interface widths "
                     f"(failed {100 * wd.failed_fraction:.0f}%)")
        ax.legend(fontsize=8)
        _save(fig, plot_dir / f"{sim_id}_width_hist.png", log)


def _plot_penetration(sim_id, plot_dir, det: FrameAnalysis, log):
    pen, slab = det.penetration, det.slab
    ext = [slab.u[0], slab.u[-1], slab.v[0], slab.v[-1]]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.3), constrained_layout=True)
    for ax, (m, t) in zip(axes[:2], [(pen.map_fB_in_A_core, "unlike fraction at A-core midplane"),
                                     (pen.map_fA_in_B_core, "unlike fraction at B-core midplane")]):
        im = ax.imshow(m.T, origin="lower", cmap="inferno", vmin=0,
                       vmax=max(0.3, np.nanpercentile(m, 99)), extent=ext, aspect="auto")
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("u"); ax.set_ylabel("v")
        fig.colorbar(im, ax=ax, shrink=0.85)
    ax = axes[2]
    ax.plot(pen.s_bins, pen.rhoA_of_s, label=r"$\rho_A(s)$", color="crimson")
    ax.plot(pen.s_bins, pen.rhoB_of_s, label=r"$\rho_B(s)$", color="navy")
    ax.axvline(0, color="k", lw=0.7)
    ax.set_xlabel(r"signed distance from interface $s$ [$r_c$] (A side $>0$)")
    ax.set_ylabel(r"density [$r_c^{-3}$]")
    ax.set_title(f"P(B->A)={pen.P_B_to_A:.3f}, P(A->B)={pen.P_A_to_B:.3f}, "
                 f"O={pen.overlap_O:.3f}", fontsize=10)
    ax.legend()
    fig.suptitle(f"{sim_id}: penetration/mixing (step {det.timestep})")
    _save(fig, plot_dir / f"{sim_id}_penetration.png", log)


def _plot_profiles(sim_id, plot_dir, det: FrameAnalysis, log):
    wd = det.width
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    ax = axes[0]
    ax.plot(wd.s_grid, wd.aligned_mean_profile, "o", ms=3, label="aligned mean profile")
    f = wd.aligned_fit
    if np.isfinite(f.get("lam", np.nan)):
        ax.plot(wd.s_grid, _tanh_model(wd.s_grid, f["a"], f["b"], f["s0"], f["lam"]),
                "-", label=f"tanh fit: $\\lambda$={f['lam']:.3f}, $R^2$={f['r2']:.4f}")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("s from local crossing [$r_c$]"); ax.set_ylabel(r"$\phi$")
    ax.set_title("locally aligned (waviness removed)")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(wd.naive_profile_w, wd.naive_profile, "-", color="gray")
    nf = wd.naive_fit
    lam_txt = f"{nf['lam']:.3f}" if np.isfinite(nf.get("lam", np.nan)) else "n/a"
    ax.set_xlabel("w [$r_c$]"); ax.set_ylabel(r"$\langle\phi\rangle_{u,v}$")
    ax.set_title(f"naive lateral average: apparent $\\lambda$={lam_txt} "
                 f"(broadening x{wd.waviness_broadening_ratio:.2f})")
    fig.suptitle(f"{sim_id}: interface profile — width vs waviness separation")
    _save(fig, plot_dir / f"{sim_id}_profiles.png", log)


def _plot_height_spectra(sim_id, plot_dir, det: FrameAnalysis, log):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), constrained_layout=True)
    ax = axes[0]
    for j, wv in enumerate(det.waviness):
        if wv.q_par.size and np.any(wv.S_iso > 0):
            m = wv.q_par > 0
            ax.loglog(wv.q_par[m], wv.S_iso[m], alpha=0.6, lw=1,
                      label=f"iface {j}" if j < 8 else None)
    ax.set_xlabel(r"$q_\parallel$ [$r_c^{-1}$]")
    ax.set_ylabel(r"$S_h(q_\parallel)$")
    ax.set_title("isotropic height power spectra (Hann-windowed)")
    ax.legend(fontsize=7, ncols=2)
    ax = axes[1]
    for j, wv in enumerate(det.waviness):
        if wv.Cr_r.size:
            ax.plot(wv.Cr_r, wv.Cr, alpha=0.6, lw=1)
    ax.axhline(np.exp(-1), color="k", ls=":", lw=0.8, label=r"$1/e$")
    ax.set_xlabel(r"lateral lag $r$ [$r_c$]")
    ax.set_ylabel(r"$C_h(r)/C_h(0)$")
    ax.set_title("height-height correlation")
    ax.legend()
    fig.suptitle(f"{sim_id}: waviness spectra (step {det.timestep})")
    _save(fig, plot_dir / f"{sim_id}_height_spectra.png", log)


def _plot_normal_angles(sim_id, plot_dir, det: FrameAnalysis, log):
    oo = det.orient_order
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), constrained_layout=True)
    ax = axes[0]
    if oo.angle_hist.size:
        ax.plot(oo.angle_hist_centers, oo.angle_hist)
        ax.axvline(oo.mean_angle_deg, color="k", ls="--",
                   label=f"mean {oo.mean_angle_deg:.1f} deg")
    ax.set_xlabel(r"local misorientation $\theta$ [deg]")
    ax.set_ylabel("area-weighted density")
    ax.set_title(f"$S_n$ = {oo.S_n:.3f}")
    ax.legend()
    ax = axes[1]
    if oo.misorientation_map.size:
        im = ax.imshow(oo.misorientation_map.T, origin="lower", cmap="plasma",
                       vmin=0, vmax=45, aspect="auto")
        fig.colorbar(im, ax=ax, shrink=0.85, label=r"$\theta$ [deg]")
    ax.set_title("projected misorientation map")
    fig.suptitle(f"{sim_id}: local-normal orientation vs $n_0$ (step {det.timestep})")
    _save(fig, plot_dir / f"{sim_id}_normal_angles.png", log)


def _plot_defect_map(sim_id, plot_dir, det: FrameAnalysis, log):
    iset, slab = det.iset, det.slab
    ext = [slab.u[0], slab.u[-1], slab.v[0], slab.v[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    cmap = ListedColormap(_DEFECT_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, 6.5), cmap.N)
    ax = axes[0]
    im = ax.imshow(iset.column_class.T, origin="lower", cmap=cmap, norm=norm,
                   extent=ext, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, ticks=range(6), shrink=0.85)
    cbar.ax.set_yticklabels([COL_CLASS_NAMES[i] for i in range(6)], fontsize=8)
    ax.set_title(f"column classification (f_bad={iset.bad_column_fraction:.3f})")
    ax.set_xlabel("u"); ax.set_ylabel("v")
    ax = axes[1]
    im = ax.imshow(iset.crossing_count.T, origin="lower", cmap="coolwarm",
                   vmin=iset.expected_crossings - 3, vmax=iset.expected_crossings + 3,
                   extent=ext, aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.85, label="crossings per column")
    ax.set_title(f"crossing count (expected {iset.expected_crossings})")
    ax.set_xlabel("u"); ax.set_ylabel("v")
    fig.suptitle(f"{sim_id}: defect maps (step {det.timestep})")
    _save(fig, plot_dir / f"{sim_id}_defect_map.png", log)


def _plot_topology(sim_id, plot_dir, det: FrameAnalysis, log):
    topo = det.topology
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True)
    ax = axes[0]
    if topo.connectivity_view is not None:
        k = topo.connectivity_view.shape[2] // 2
        im = ax.imshow(topo.connectivity_view[:, :, k].T, origin="lower",
                       cmap="tab20", interpolation="nearest")
        fig.colorbar(im, ax=ax, shrink=0.85, label="component id")
    threshold_mode = str(topo.diagnostics.get("threshold_mode", "absolute_phi"))
    ax.set_title("A-phase components at threshold 0 (mid slice)")
    rows = pd.DataFrame(topo.rows)
    for ax, col, ylab in ((axes[1], "beta0", r"components $\beta_0$"),
                          (axes[2], "chi_total", r"Euler characteristic $\chi$")):
        for phase, style in (("A", "-o"), ("B", "--s")):
            sub = rows[(rows["phase"] == phase) & np.isfinite(rows[col])]
            ax.plot(sub["phi_c"], sub[col], style, ms=4, label=f"{phase} phase")
        exp = rows["expected_components_perfect"].dropna()
        if col == "beta0" and len(exp):
            ax.axhline(float(exp.iloc[0]), color="k", ls=":", lw=0.8,
                       label="perfect lamellae")
        ax.set_xlabel("normalized plateau threshold" if threshold_mode == "normalized_plateaus"
                      else r"threshold $\phi_c$")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=8)
    fig.suptitle(f"{sim_id}: periodic topology vs threshold (step {det.timestep})")
    _save(fig, plot_dir / f"{sim_id}_topology.png", log)


def _plot_timeseries(sim_id, plot_dir, frame_df: pd.DataFrame, log):
    cols = [("waviness_rms", "RMS waviness [$r_c$]"),
            ("width_lam_corrected_mean", r"width $\lambda$ [$r_c$]"),
            ("bad_column_fraction", "bad-column fraction"),
            ("spacing_mean", "mean spacing [$r_c$]"),
            ("S_n", r"$S_n$"),
            ("P_A_to_B", r"$P_{A\to B}$")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.5), constrained_layout=True)
    for ax, (col, ylab) in zip(axes.ravel(), cols):
        if col in frame_df:
            ax.plot(frame_df["timestep"], frame_df[col], "o-", ms=3, lw=0.8)
        ax.set_xlabel("timestep"); ax.set_ylabel(ylab)
    fig.suptitle(f"{sim_id}: per-frame metrics (frames may be correlated)")
    _save(fig, plot_dir / f"{sim_id}_timeseries.png", log)
