"""Unit-test-like sanity checks run after each simulation's analysis.

These checks detect obvious errors (wrong bead type assignments, empty windows,
pathological q_star values, failed S_tot consistency) that would silently
corrupt downstream results. They do not constitute scientific validation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List


def run_sanity_checks(
    result: Dict[str, Any],
    cfg: Dict[str, Any],
) -> List[str]:
    """Run all sanity checks and return a list of error/warning strings.

    Returns [] if all checks pass. Each failure begins with 'ERROR' or 'WARN'.
    """
    issues: List[str] = []

    # 1. Valid bead type selection (both A and B must be present)
    if not result.get("has_A_beads", True):
        issues.append(
            f"ERROR: A bead type (type_id={cfg['A_type_id']}) not found in trajectory. "
            "Check A_type_id in config."
        )
    if not result.get("has_B_beads", True):
        issues.append(
            f"ERROR: B bead type (type_id={cfg['B_type_id']}) not found in trajectory. "
            "Check B_type_id in config."
        )

    # 2. Non-empty accepted window
    n_accepted = result.get("n_accepted_frames", 0)
    if n_accepted == 0:
        issues.append("ERROR: Accepted window contains zero frames. No averaging was performed.")
    elif n_accepted < 3:
        issues.append(
            f"WARN: Only {n_accepted} frame(s) in the accepted window. "
            "Statistical uncertainty in averaged S(q) will be large."
        )

    # 3. Fallback-window flag
    if result.get("fallback_used", False):
        issues.append(
            "WARN: Stationary-window heuristic did not converge; a fallback window was used. "
            "Inspect window_diagnostics.csv and diagnostic plots before trusting q_star."
        )

    # 4. q_star is required only when a statistically significant peak exists.
    # A disordered spectrum is allowed to have no q_star; older versions
    # incorrectly forced one by choosing the largest noise bin.
    q_star = result.get("q_star_sq", float("nan"))
    peak_detected = bool(result.get("peak_detected", False))
    if peak_detected and not math.isfinite(q_star):
        issues.append(
            "ERROR: peak_detected is true but q_star is NaN or Inf."
        )
    elif peak_detected and q_star <= 0.0:
        issues.append(f"ERROR: q_star = {q_star:.4f} <= 0. This is unphysical.")

    # 5. q_star not at the boundary of the searched q range
    if peak_detected and math.isfinite(q_star) and q_star > 0:
        if result.get("q_star_at_lower_boundary", False):
            q_lo = result.get("q_peak_search_min_resolved", cfg.get("q_peak_search_min"))
            issues.append(
                f"WARN: q_star ({q_star:.4f}) is within 2 bins of q_peak_search_min ({q_lo}). "
                "The true peak may lie below the search range — consider decreasing q_peak_search_min "
                "(note it cannot go below the box's fundamental reciprocal spacing)."
            )
        if result.get("q_star_at_upper_boundary", False):
            q_hi = result.get("q_peak_search_max_resolved", cfg.get("q_compute_max", 4.0))
            issues.append(
                f"WARN: q_star ({q_star:.4f}) is within 2 bins of the peak-search upper bound ({q_hi}). "
                "The true peak may lie above — consider increasing q_peak_search_max or q_compute_max."
            )

    # 6. Consistent particle count across frames
    if not result.get("n_frames_consistent", True):
        issues.append(
            "WARN: Particle count changed between sampled frames. "
            "Set allow_variable_natoms: true in config to suppress this check."
        )

    # 7. S_tot consistency check (S_tot ≈ S_AA + S_BB + 2*S_AB)
    s_tot_err = result.get("s_tot_check_err", 0.0)
    if math.isfinite(s_tot_err) and s_tot_err > 1e-4:
        issues.append(
            f"WARN: S_tot consistency check failed (max relative error = {s_tot_err:.2e}). "
            "Expected ~float32 precision (<1e-4 relative to the partials); "
            "a large value indicates a bug in partial-SF computation."
        )

    # 8. FFT cross-check (if available)
    q_star_fft = result.get("q_star_fft_mean", float("nan"))
    tol = cfg.get("fft_crosscheck_tol", 0.15)
    if math.isfinite(q_star) and math.isfinite(q_star_fft) and q_star > 0 and q_star_fft > 0:
        rel_diff = abs(q_star - q_star_fft) / q_star
        if rel_diff > tol:
            issues.append(
                f"WARN: publishable Sq q_star ({q_star:.4f}) and diagnostic "
                f"q_star ({q_star_fft:.4f}) "
                f"differ by {rel_diff:.1%} > tolerance {tol:.0%}. "
                "Check grid resolution and q range settings; large single-frame noise is normal."
            )

    # 9. Statistical support of the q_star bin: how many reciprocal vectors
    # contribute to the bin that holds the reported peak value.
    n_kvec_bin = result.get("n_kvec_in_q_star_bin")
    if n_kvec_bin is not None and n_kvec_bin >= 0 and n_kvec_bin < 10:
        issues.append(
            f"WARN: The q_star bin contains only {n_kvec_bin} reciprocal vector(s). "
            "The peak value is an average over very few k-vectors and is statistically "
            "noisy — consider a wider q_bin_width or more accepted frames."
        )

    # 10. Bin width much finer than the box's fundamental reciprocal spacing:
    # the plotted curve then looks better-resolved than the underlying discrete
    # q-vector set actually is (many empty bins between populated shells).
    q_bin_width = result.get("q_bin_width", float("nan"))
    dq_min = result.get("dq_min", float("nan"))
    if math.isfinite(q_bin_width) and math.isfinite(dq_min) and q_bin_width < dq_min / 10.0:
        issues.append(
            f"WARN: q_bin_width ({q_bin_width:.5f}) is less than one-tenth of the fundamental "
            f"reciprocal spacing ({dq_min:.5f}). The S(q) curve will appear more finely "
            "resolved than the discrete reciprocal-vector set supports (many empty bins)."
        )

    # 11. Box dimensions varying over the accepted window: the q-grid was
    # resolved from the first frame's box, so a drifting box invalidates it.
    box_var = result.get("box_dims_max_rel_variation", 0.0)
    if math.isfinite(box_var) and box_var > 0.005:
        issues.append(
            f"WARN: Box dimensions vary by {box_var:.2%} over the accepted window. "
            "The q-grid (resolved from the first frame) and the per-frame reciprocal "
            "lattices do not match across frames; q_star and L0 are smeared accordingly."
        )

    return issues


def format_check_report(issues: List[str], sim_id: str) -> str:
    if not issues:
        return f"[{sim_id}] All sanity checks passed."
    lines = [f"[{sim_id}] Sanity check results:"]
    for issue in issues:
        lines.append(f"  {issue}")
    return "\n".join(lines)
