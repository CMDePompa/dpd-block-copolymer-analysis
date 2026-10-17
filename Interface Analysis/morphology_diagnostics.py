"""Diagnostic plots for curved and network morphology branches."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def make_morphology_plots(sim_id, plot_dir, detail, frame_df, profile_df) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    morphology = detail.domains.morphology
    interface_level = float(detail.surface.diagnostics.get("interface_level", 0.0))

    fields = [detail.fields.rho_A, detail.fields.rho_B, detail.comp.phi]
    labels = [r"$\rho_A$", r"$\rho_B$", r"$\phi$"]
    cmaps = ["Blues", "Oranges", "RdBu_r"]
    fig, axes = plt.subplots(3, 3, figsize=(10, 9), constrained_layout=True)
    for row, (arr, label, cmap) in enumerate(zip(fields, labels, cmaps)):
        for axis in range(3):
            index = arr.shape[axis] // 2
            image = np.take(arr, index, axis=axis).T
            vmax = None if label != r"$\phi$" else max(abs(float(np.nanmin(arr))),
                                                        abs(float(np.nanmax(arr))))
            im = axes[row, axis].imshow(
                image, origin="lower", cmap=cmap,
                vmin=(-vmax if vmax is not None else None), vmax=vmax,
            )
            if label == r"$\phi$":
                axes[row, axis].contour(
                    image, levels=[interface_level], colors="k", linewidths=0.6,
                )
            axes[row, axis].set_title(f"{label}, {['x','y','z'][axis]} midplane")
            fig.colorbar(im, ax=axes[row, axis], shrink=0.75)
    fig.suptitle(f"{sim_id}: {morphology} density/composition slices")
    _save(fig, plot_dir / f"{sim_id}_field_slices.png")

    surf = detail.surface
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    axes[0].hist(surf.mean_curvature[np.isfinite(surf.mean_curvature)], bins=60,
                 weights=surf.face_areas[np.isfinite(surf.mean_curvature)], density=True)
    axes[0].set(xlabel=r"mean curvature $H$ [$r_c^{-1}$]", ylabel="area density")
    axes[1].hist(surf.gaussian_curvature[np.isfinite(surf.gaussian_curvature)], bins=60,
                 weights=surf.face_areas[np.isfinite(surf.gaussian_curvature)], density=True)
    axes[1].set(xlabel=r"Gaussian curvature $K$ [$r_c^{-2}$]", ylabel="area density")
    eig = surf.normal_tensor_eigenvalues
    axes[2].bar(["min", "mid", "max"], eig)
    axes[2].axhline(1 / 3, color="k", ls="--", lw=1, label="isotropic")
    axes[2].set(ylabel=r"eigenvalue of $\langle nn\rangle_A$", ylim=(0, 1))
    axes[2].legend(fontsize=8)
    fig.suptitle(
        f"{sim_id}: periodic interface geometry; A/V={surf.area_per_volume:.3g}, "
        f"negative-K area={surf.K_negative_area_fraction:.2%}"
    )
    _save(fig, plot_dir / f"{sim_id}_curvature_and_normals.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    s = profile_df["s"].to_numpy()
    phi = profile_df["phi_mean_frames"].to_numpy()
    phi_sd = profile_df["phi_std_frames"].to_numpy()
    axes[0].plot(s, phi, color="black", label=r"$\phi$")
    axes[0].fill_between(s, phi - phi_sd, phi + phi_sd, color="black", alpha=0.18,
                         label="SD over frames")
    axes[0].axvline(0, color="gray", ls="--", lw=1)
    axes[0].set(xlabel=r"local normal distance $s$ [$r_c$]", ylabel="composition",
                title="intrinsic composition profile")
    axes[0].legend(fontsize=8)
    for key, label, color in (("rho_A", r"$\rho_A$", "#1f77b4"),
                              ("rho_B", r"$\rho_B$", "#d95f02")):
        axes[1].plot(s, profile_df[f"{key}_mean_frames"], label=label, color=color)
    axes[1].axvline(0, color="gray", ls="--", lw=1)
    axes[1].set(xlabel=r"local normal distance $s$ [$r_c$]", ylabel="number density",
                title="interface-normal density profiles")
    axes[1].legend()
    fig.suptitle(f"{sim_id}: {morphology} interface-normal profiles")
    _save(fig, plot_dir / f"{sim_id}_normal_profiles.png")

    keys = [
        ("surface_area_per_volume", "area / volume"),
        ("surface_H_rms", r"RMS $H$"),
        ("surface_K_negative_area_fraction", "negative-K area fraction"),
        ("normal_width_lam_mean", r"intrinsic $\lambda$"),
        ("domain_component_count", "minority components"),
        ("overlap_O", "density overlap"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)
    for ax, (key, label) in zip(axes.ravel(), keys):
        ax.plot(frame_df["timestep"], frame_df[key], "-o", ms=3)
        ax.set(xlabel="timestep", ylabel=label)
    fig.suptitle(f"{sim_id}: {morphology} interface time series")
    _save(fig, plot_dir / f"{sim_id}_time_series.png")

    rows = detail.domains.component_rows
    if rows:
        radii = np.asarray([r["equivalent_sphere_radius"] for r in rows], dtype=float)
        core = np.asarray([r["maximum_core_distance"] for r in rows], dtype=float)
        asph = np.asarray([r["asphericity"] for r in rows], dtype=float)
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)
        axes[0].hist(radii[np.isfinite(radii)], bins="auto")
        axes[0].set(xlabel="equivalent-sphere radius", ylabel="components")
        axes[1].hist(core[np.isfinite(core)], bins="auto")
        axes[1].set(xlabel="maximum core distance", ylabel="components")
        axes[2].hist(asph[np.isfinite(asph)], bins="auto")
        axes[2].set(xlabel="component asphericity", ylabel="components")
        fig.suptitle(f"{sim_id}: last-frame minority-domain components")
        _save(fig, plot_dir / f"{sim_id}_domain_components.png")
