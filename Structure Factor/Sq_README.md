# Structure-factor analysis pipeline

This pipeline analyzes the selected A/B pair in LAMMPS block-copolymer
trajectories. A trajectory may contain additional bead types, but only the
types selected by `A_type_id` and `B_type_id` enter the spectra; normalization
uses `N_A + N_B`. It does not calculate correlations involving other species
or a complete multicomponent structure-factor matrix. It supports a two-stage
campaign:

1. Run the fast CIC calculation on every trajectory.
2. Inspect the spectra and the stationarity checks.
3. Re-run selected ordered trajectories with the exact calculation.

The central output is a composition structure factor, S(q). A peak at a
finite q means that the A/B composition repeats at a characteristic distance.
If a significant peak is present,

    L0 = 2*pi/q_star

is the corresponding real-space period.

Version 4 never invents q_star. If the spectrum does not contain a significant
interior local maximum, the output explicitly says peak_detected=false and
leaves q_star and L0 undefined.

## What the pipeline can and cannot establish

The pipeline can show that measured quantities stopped changing over the
sampled interval. It writes this as spectral stationarity.

Stationarity is consistent with either:

- thermodynamic equilibrium, or
- a long-lived metastable state.

A finite trajectory cannot distinguish those possibilities by itself. The
recommended paper language is therefore:

> The measured structure became stationary over the sampled interval,
> consistent with an equilibrium or long-lived metastable state.

The pipeline does not automatically assign a morphology and does not claim
that a stationary structure is the globally stable phase.

## The analysis in plain language

For every saved simulation frame:

1. Read the A- and B-bead coordinates.
2. Compute inexpensive grid-based diagnostics used for time-window selection.
3. Compute the requested structure factor if the frame belongs to the S(q)
   averaging window.
4. Average the accepted frame spectra.
5. Divide the accepted interval into blocks and compare the complete spectrum
   between blocks.
6. Look for a statistically meaningful finite-q local maximum.
7. Write spectra, uncertainties, validation records, stationarity records,
   summaries, and diagnostic plots.

## CIC and exact engines

Select the engine in YAML:

~~~yaml
sq_method: cic
~~~

or

~~~yaml
sq_method: exact
~~~

### CIC

CIC deposits particles onto a periodic three-dimensional mesh, performs an
FFT, and removes the known CIC assignment window. It is fast enough for
campaign screening.

CIC is an approximation. The pipeline therefore compares CIC against a direct
particle sum at low q on frames sampled across the beginning, middle, and end
of the actual accepted S(q) window.

For a spectrum with a significant peak, validation checks the peak bin and
peak intensity. For a spectrum without a significant peak, it compares the
complete low-q spectrum by relative RMS error. A disordered spectrum is not
forced to have a peak merely for validation.

### Exact

The exact engine evaluates

    F_A(q) = sum over A beads of exp(i q dot r_j)
    F_B(q) = sum over B beads of exp(i q dot r_j)

at every reciprocal-lattice vector inside the configured q range. It uses the
complete lattice, including zero components and mixed signs such as
(1, -1, 2). It performs no spatial gridding or reciprocal-shell subsampling.

Exact is much more expensive and is intended for selected reruns.

## Structure-factor definitions

Every partial spectrum is normalized by N = N_A + N_B:

    S_AA  = <|F_A|^2>/N
    S_BB  = <|F_B|^2>/N
    S_AB  = <Re(F_A F_B*)>/N
    S_tot = <|F_A + F_B|^2>/N

The raw composition contrast is

    S_comp = <|F_A - F_B|^2>/N
           = S_AA + S_BB - 2*S_AB

This remains useful as a scattering-like measure.

### Composition-normalized concentration spectrum

If the global A/B fractions change between systems, raw S_comp amplitudes are
not directly composition-neutral. Let

    x_A = N_A/N
    x_B = N_B/N

The pipeline also reports

    S_cc = x_B^2*S_AA + x_A^2*S_BB - 2*x_A*x_B*S_AB

and

    S_cc_normalized = S_cc/(x_A*x_B)

The normalized curve removes the trivial amplitude change produced solely by
changing the global A/B fractions. Peak detection is performed on
S_cc_normalized. Both raw and normalized curves are saved; neither is hidden.

## No forced q_star

A q_star is reported only when an interior local maximum passes transparent
significance tests:

~~~yaml
peak_min_height_over_median: 1.5
peak_min_relative_prominence: 0.10
peak_min_occupied_bins: 5
~~~

The candidate peak must:

- be a local maximum, not an endpoint;
- be at least peak_min_height_over_median times the median spectrum in the
  search range;
- have relative prominence above peak_min_relative_prominence; and
- be supported by enough occupied q bins.

If no candidate passes:

~~~text
peak_detected = false
q_star = NaN
L0 = NaN
peak_detection_method = no_peak
~~~

This is expected behavior for a diffuse or disordered spectrum. Inspect the
log-S(q) plot before relaxing the thresholds.

The code does not turn this peak test into a manufactured morphology score.
The spectra and the individual peak diagnostics remain available for notebook
analysis.

## Choosing the S(q) averaging window

There are three mutually exclusive choices.

### Automatic stationary window

~~~yaml
sq_window_steps: null
sq_window_start_step: null
sq_window_end_step: null
~~~

The detector divides the post-burn-in trajectory into timestep blocks and
requires a stationary run at the end.

### Fixed trailing window

~~~yaml
sq_window_steps: 100000
~~~

Frames with timestep greater than or equal to final_timestep minus 100000 are
averaged.

### Explicit window

~~~yaml
sq_window_start_step: 3000000
sq_window_end_step: 3100000
~~~

This is the preferred way to make an exact rerun use precisely the same window
as a CIC screening run. Copy the accepted bounds from accepted_windows.json.

The final-timestep reader verifies that the final atom block is complete. A
truncated frame from an interrupted simulation is ignored.

## Time-window diagnostics

The inexpensive diagnostic constructs a local composition field from CIC
densities. The configured phi mode is now honored:

~~~yaml
phi_mode: smooth
phi_smooth_sigma: 1.0
~~~

Available modes:

- zero: local composition in occupied voxels; empty voxels are zero.
- nan_mean: empty voxels receive the mean occupied composition.
- smooth: periodically smooth A and B densities before forming the local
  composition.

The default stationarity gates are phi_rms and thermo:

~~~yaml
stationarity_metrics:
  - phi_rms
  - thermo
~~~

q_star_fft and peak_height_fft remain available as descriptive measurements.
q_star_fft is reciprocal-grid-quantized, while a short-block peak-height CV can
miss slow coarsening between blocks. The post-selection S(q) plateau check
therefore includes an explicit long-horizon peak-intensity drift test.

If thermo is a gate, every requested thermo column must exist and have enough
measurements in the block. Missing columns cannot silently pass.

For trajectories without matching out files, remove thermo from the gate:

~~~yaml
stationarity_metrics:
  - phi_rms
~~~

## Full-spectrum stationarity

After selecting the S(q) averaging frames, the pipeline divides those frames
into timestep blocks and calculates a complete normalized concentration
spectrum for every block.

It compares consecutive spectra using

    RMS of log(1 + S_block(q)) - log(1 + S_previous(q))

It additionally compares spectra from the two ends of the whole candidate
window. When a significant peak exists, it fits the normalized intensity at
that q bin across every block. This last check catches slow strengthening or
coarsening that can look quiet inside each short block.

Configure:

~~~yaml
sq_stationarity_min_blocks: 3
sq_spectral_distance_tolerance: 0.10
sq_peak_intensity_slope_tolerance: 0.02
~~~

Outputs:

- sq_block_spectra.csv: every block-averaged spectrum;
- sq_block_metrics.csv: spectral distance and integrated intensity by block;
- sq_stationarity.json: thresholds and the final stationarity verdict;
- block-based standard errors in sq_composition.csv.

The field sq_spectral_stationary is true, false, or blank when insufficient
blocks exist. `stationarity_supported_by_all_checks` is true only when both the
candidate time window and the S(q) plateau check pass. These are stationarity
checks, not equilibrium certificates.

## Strict failure handling

Every requested S(q) frame must succeed.

If one requested averaging frame fails:

- analysis_complete is false;
- numerical certification is false;
- the failed timestep is recorded;
- the command exits nonzero;
- an error summary overwrites any older per-simulation success summary.

The pipeline never silently averages only the successful subset and calls the
result certified.

Numerical validity, stationarity, and peak detection are separate fields:

- analysis_complete
- numerical_validation_passed
- sq_spectral_stationary
- peak_detected

## Main configuration example

~~~yaml
sq_method: cic

input_dir: /path/to/trajectories
output_dir: /path/to/Sq_pipeline_output_v4/cic/a12_30
traj_suffix: .prod.lammpstrj
out_prefix: out.

frame_stride: 1
analysis_start_step: null
burn_in_fraction: 0.2

block_size_steps: 25000
min_blocks_accepted: 4
metric_slope_tolerance: 0.02
metric_cv_tolerance: 0.05
fallback_fraction: 0.30

stationarity_metrics:
  - phi_rms
  - thermo

thermo_stationarity_cols:
  - PotEng
  - TotEng
  - Press
  - Temp

sq_window_steps: 100000
sq_window_start_step: null
sq_window_end_step: null

q_compute_min: auto
q_compute_max: 4.0
q_bin_width: auto
q_bins: auto
q_peak_search_min: auto
q_peak_search_max: null

peak_min_height_over_median: 1.5
peak_min_relative_prominence: 0.10
peak_min_occupied_bins: 5

sq_stationarity_min_blocks: 3
sq_spectral_distance_tolerance: 0.10
sq_peak_intensity_slope_tolerance: 0.02

sq_grid: 128
exact_validation_frames: 3
exact_validation_q_max: 1.0
exact_validation_peak_rel_tol: 0.10

A_type_id: 1
B_type_id: 2

run_fft_diagnostic: true
fft_grid: 64
phi_mode: smooth
phi_smooth_sigma: 1.0

save_per_frame_spectra: false
plots: true
allow_variable_natoms: false
~~~

## Running the pipeline

Install Python 3.10 or later and:

~~~bash
python -m pip install -r requirements.txt
~~~

Create an annotated template:

~~~bash
python run_pipeline.py --make-template my_config.yaml
~~~

Check dependencies:

~~~bash
python run_pipeline.py --version
~~~

Run a two-frame preflight:

~~~bash
python run_pipeline.py --config my_config.yaml --test-mode
~~~

Run a batch:

~~~bash
python run_pipeline.py --config my_config.yaml
~~~

Run trajectories in parallel:

~~~bash
python run_parallel.py --config my_config.yaml -j 4
~~~

### LSF/HPC submission

The `submit_lsf.sh` script initializes Conda explicitly because LSF launches a
non-interactive shell. By default they activate the shared environment prefix
`/usr/local/usrapps/pasquinell/md_env`. You
can override that without editing the script:

~~~bash
CONDA_ENV=/path/to/my_analysis_env bsub < submit_lsf.sh
~~~

If your cluster exposes Conda through a module, load that module before
submission. If Conda is installed in a nonstandard location, set
`CONDA_BASE=/path/to/miniconda3` in the job environment. The scripts use
`PYTHON_BIN` when set, otherwise the Python executable from the activated
environment. Environment setup is now fail-fast, so a missing activation cannot
silently fall through to an incompatible system Python.

Run one exact rerun:

~~~bash
python run_pipeline.py --config my_config.yaml --sq-method exact \
  --traj-file trajectory.prod.lammpstrj \
  --output-dir /path/to/Sq_pipeline_output_v4/exact/a12_30
~~~

Use different output roots for CIC and exact results.

## Output files

Per simulation:

| File | Meaning |
|---|---|
| sq_composition.csv | raw S_comp, S_cc, normalized S_cc, and block uncertainties |
| sq_partial.csv | S_AA, S_BB, S_AB, and S_tot |
| sq_block_spectra.csv | block-averaged spectra through the accepted interval |
| sq_block_metrics.csv | full-spectrum convergence measurements |
| sq_stationarity.json | explicit stationarity thresholds and result |
| sq_normalization.json | formulas, fractions, method, and peak rules |
| sq_resolution.json | box, reciprocal resolution, bins, and vector counts |
| exact_direct_validation.json | numerical validation details |
| accepted_windows.json | detector window and actual averaged timesteps |
| frame_metrics.csv | inexpensive time-dependent diagnostics |
| window_diagnostics.csv | stationarity measurements by detector block |
| analysis_summary.csv | one-row machine-readable summary |

When requested:

- thermo_timeseries.csv
- job_metadata.json
- sq_per_frame.csv
- diagnostic_plots/

At the output root:

- analysis_summary.csv
- all_thermo_timeseries.csv
- all_job_status.csv
- run_config_TIMESTAMP.yaml
- plots/SIM_ID_sq_log.png

The plots directory contains one 200-dpi log-y S(q) screening plot per
trajectory. It overlays raw composition contrast and composition-normalized
concentration S(q), and explicitly annotates whether a significant peak was
detected. The per-simulation `diagnostic_plots/*_composition_sq.png` plot is
also log-y. Window plots display thermo data only over the trajectory interval,
so earlier ramps and annealing stages do not compress the production data.

After a plot-only update, saved CSV outputs can be replotted without rereading
the large trajectory files:

~~~bash
python refresh_saved_plots.py --output-dir /path/to/output --config config.yaml
~~~

This also writes a non-destructive `stationarity_reassessment.csv` using the
current S(q) plateau rules; it does not overwrite the original analysis rows.

## Recommended result checks

Before using a trajectory result, require:

1. analysis_complete is true.
2. numerical_validation_passed is true.
3. The requested and averaged frame counts match.
4. The accepted window is the intended window.
5. stationarity_supported_by_all_checks and sq_spectral_stationary have been reviewed.
6. Fallback-window warnings have been reviewed.
7. The log-S(q) plot agrees with peak_detected.
8. CIC grid convergence has been checked on representative ordered,
   boundary, and disordered trajectories.
9. q_star and L0 are interpreted only when peak_detected is true.

For a disordered result, an undefined q_star is correct and expected.

## Assumptions

- Periodic orthorhombic simulation boxes.
- Fixed box dimensions across an analyzed spectrum.
- A and B are identified by integer type IDs.
- q values belong to the box reciprocal lattice.
- Time-adjacent frames can be correlated; reported block uncertainties should
  be interpreted using a scientifically chosen block size.
- Radial S(q) discards directional information. Use real-space snapshots or
  directional analysis when assigning a detailed morphology.
