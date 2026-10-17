# Interface Analysis V2

V2 analyzes interfaces between the bead types selected by `A_type_id` and
`B_type_id` in LAMMPS/DPD trajectories with lamellar, spherical, cylindrical,
or gyroid morphology. It can read frames with additional types, but its
density and composition fields use only A and B. Some bead-level mixing
fractions still count all particles, and the optional chain-region profiles
reject extra types. Results from a mixture with more than two species are
therefore not a complete or consistently filtered multicomponent interface
analysis. The user declares the morphology after visual and S(q) validation;
this code deliberately does **not** classify, relabel, or choose a morphology.

The lamellar route uses height-field analysis. Nonlamellar routes use periodic
three-dimensional surfaces and local-normal profiles.

## Quick start

From `Post-Processing Analysis/Interface Analysis`:

```powershell
python run_interface.py --make-template my_interface_v2.yaml
python run_interface.py --config my_interface_v2.yaml --test-mode
python run_interface.py --config my_interface_v2.yaml
```

The essential config choice is:

```yaml
morphology: gyroid        # lamella | sphere | cylinder | gyroid
minority_phase: auto      # auto | A | B; used for spheres/cylinders

input_dir: ./traj_files
output_dir: ./interface_v2_output
sq_results_dir: ../Structure Factor/output

target_voxel_size: 0.5
smoothing_sigma: 0.75
interface_level: 0.0
surface_normal_samples: 2048
normal_profile_points: 65
normal_profile_halfwidth: auto
```

Use one config/run per declared morphology. `--morphology sphere` can override
the YAML for a deliberate one-off run. Unknown config keys are errors.

## What is shared across morphologies

Each frame is deposited on an orthorhombic periodic grid by cloud-in-cell and
smoothed with a periodic Gaussian. The composition field is

\[
\phi(\mathbf r)=\frac{\rho_A(\mathbf r)-\rho_B(\mathbf r)}
                       {\rho_A(\mathbf r)+\rho_B(\mathbf r)}.
\]

The nominal A/B dividing surface is `phi = interface_level` (normally zero,
equal local A and B density). Grid size, actual voxel dimensions, smoothing,
invalid-density voxels, and every automatic resolution rule are recorded.

S(q) is read-only. Its q-star supplies the characteristic length
`L0 = 2*pi/q_star` and accepted production window when available. If absent,
V2 reports that it used the strongest composition-field FFT magnitude as a
length-scale fallback. Neither source is used to select the morphology.

## Morphology routing

| Declared morphology | Interface representation | Main geometry |
|---|---|---|
| `lamella` | V1 oblique columns and height fields | orientation, waviness spectrum, spacing, excess area, slope, lamellar intrinsic width, defects, penetration |
| `sphere` | periodic 3-D phi isosurface | sphere count, equivalent and V/A radii, core distance, asphericity, nearest-neighbor distance, wrapping warning |
| `cylinder` | periodic 3-D phi isosurface | cylinder V/A radius, normal-tensor axis and alignment, core distance, component continuity/wrapping |
| `gyroid` | periodic 3-D phi isosurface | area/volume, H and K distributions, negative-K area fraction, Euler characteristic, three-direction network wrapping |

Lamellar orientation/resampling/height modules are explicitly skipped for the
three nonlamellar routes. This matters because sphere, cylinder, and gyroid
surfaces contain overhangs or branches and generally cannot be represented by
one single-valued height function.

## Nonlamellar instantaneous surface

V2 extracts one fundamental periodic cell of the configured phi isosurface by
marching cubes with a positive-side periodic ghost layer. Face normals follow
`grad(phi)` and point toward the A-rich phase. It reports:

- total area and area/box volume;
- area-weighted signed mean curvature H, mean |H|, and RMS H;
- area-weighted Gaussian curvature K and the fraction of area with K < 0;
- the area-weighted normal tensor and its eigenvalues;
- `integral(K)dA/(4*pi)` as a mesh/level-set Gauss-Bonnet estimate of the
  Euler characteristic of the enclosed solid phase.

For an ideal sphere, |H| = 1/R and K = 1/R². For an ideal cylinder,
|H| = 1/(2R) and K = 0. Curvature is calculated from derivatives of the
smoothed level-set field, not from raw triangle angles. The sign of H depends
on which phase is inside; compare `surface_H_abs_mean` or `surface_H_rms`
unless the phase convention is central to the question.

Curvature is particularly sensitive to voxel size and smoothing. All systems
in a publication comparison must use the same physical field settings, and a
smaller/finer and larger/coarser sensitivity pair should be checked on
representative systems.

## Intrinsic profiles and width

For sphere, cylinder, and gyroid frames, surface triangles are sampled
systematically in cumulative area. At each selected surface location, V2
interpolates phi, rho_A, and rho_B along the local normal, with `s=0` on the
instantaneous interface and positive s toward A. Profiles are centered before
pooling, so translations, capillary wandering, and large-scale shape do not
broaden the reported intrinsic profile.

Two complementary width families are reported:

1. Local tanh fits, including the spatial distribution of lambda and the
   corresponding 10–90% width. This preserves continuity with V1.
2. The second moment of `|d<phi>/ds|` and an error-function fit. The fitted
   Gaussian sigma is an independent width estimator and can be deconvolved
   from Gaussian smoothing in quadrature.

The error-function/moment result is not silently substituted for the tanh
result. Disagreement is useful: it flags non-tanh profile shape, insufficient
bulk plateaus, nearby second interfaces, defects, or resolution sensitivity.

If `density_profile_enabled: true`, local-normal profiles also separate
A beads in the initial chain segment from A and B beads in the remaining
fixed blocks. This requires a uniform chain length and known bead order.

## Domains and topology

For spheres and cylinders, `minority_phase: auto` chooses the less abundant
thresholded phase in that frame. Prefer explicit `A` or `B` for a publication
campaign so the convention cannot switch near equal composition.

Connected components and wrapping are periodic-aware. A component has three
Cartesian winding flags. The reported `wrap_rank_cartesian` is diagnostic, not
a full algebraic winding rank; a tilted tube can wind through multiple
Cartesian directions. Shape inertia/asphericity is intentionally omitted for
wrapping objects because no unique one-cell unwrapping exists.

Two distinct topology measurements are retained:

- `domain_interface_level_chi` is the Euler characteristic at the actual
  physical interface level used to create the mesh.
- `topology_by_threshold.csv` sweeps normalized plateau thresholds for robust
  comparisons when the measured bulk phi plateaus differ between systems.

Their threshold meanings are not interchangeable. The surface
Gauss-Bonnet estimate is cross-checked only against
`domain_interface_level_chi`.

## Statistical meaning

Analyze multiple equilibrated frames. V2 writes every per-frame measurement,
then reports mean, sample standard deviation, and (for selected time series)
an integrated-autocorrelation diagnostic/effective frame count. Consecutive
frames and thousands of surface samples from one frame are **not** independent
simulations. They characterize temporal and spatial variability within a run;
replicate simulations remain the independent units for inferential statistics.

A practical publication workflow is:

1. use S(q) stationarity to choose an equilibrated window;
2. sample frames farther apart than the fast local relaxation time;
3. inspect per-frame traces and estimated effective counts;
4. average within each independent simulation;
5. compare conditions using replicate-level values when replicates exist.

## Output contract

Every run produces `interface_comparison.csv` and a compact headline table.
Each simulation has its own folder.

Nonlamellar output files:

| File | Contents |
|---|---|
| `interface_per_frame.csv` | common, curvature, width, mixing, domain, and topology headline metrics |
| `interface_summary.csv/json` | temporal summaries, effective-count diagnostics, provenance |
| `normal_profiles_per_frame.csv` | surface-mean and surface-SD profile for every analyzed frame |
| `normal_profiles.csv/.npz` | profile mean/SD across frames and full per-frame arrays |
| `domain_components_per_frame.csv` | one row per periodic minority component |
| `topology_by_threshold.csv` | A- and B-phase periodic topology threshold sweep |
| `interface_surface_last.npz` | last-frame vertices, faces, normals, H, and K when maps are enabled |
| `interface_maps.npz` | last-frame phi, densities, phase mask, and labels when maps are enabled |
| `interface_resolution.json` | resolved grid, length scale, routing, and method notes |
| `interface_warnings.json` | summary and per-frame scientific warnings |
| `interface_parameters.json` | exact config hash and software provenance |
| `frame_read_report.json` | selected, parsed, and failed frame accounting |

The lamellar route preserves the V1 output names and meanings, with
`morphology: lamella` added to V2 summaries.

## What to emphasize first

For a compact order–order-transition paper, start with:

- all morphologies: intrinsic local-normal width, mixing fraction, density
  overlap, and interface area/volume;
- spheres: number, radius distribution, and nearest-neighbor spacing;
- cylinders: radius, axis alignment, and continuity;
- gyroids: area/volume, negative-K area fraction, and Euler
  characteristic/wrapping;
- lamellae: spacing, waviness, excess area, intrinsic width, and bad-column
  fraction.

Treat the remaining columns as diagnostics unless a mechanistic result makes
them central.

## Limitations and safeguards

- Boxes must be orthorhombic; triclinic LAMMPS boxes are not supported.
- Marching-cubes area and derivative curvature are properties of the chosen
  coarse-grained field. Report voxel and smoothing settings.
- Surface samples are systematically area-distributed, not independent.
- The automatic cylinder axis assumes a reasonably aligned cylinder ensemble;
  use an explicit axis or interpret only scalar metrics for polydomain samples.
- A gyroid warning is issued when the chosen phase is not triply wrapping, but
  the code does not call the structure “not a gyroid.”
- Sphere/cylinder warnings similarly diagnose wrapping or poor alignment
  without overriding the configured target.
- `--calibrate` retains V1's lamellar calibration. For nonlamellar phases,
  perform grid/smoothing sensitivity runs with separate configs.

## Verification

The test suite includes analytic level-set regressions for:

- sphere area, curvature, normal isotropy, radius, components, and chi;
- periodic cylinder area, curvature, axis, radius, wrapping, and chi;
- saddle-dominated, triply connected gyroid topology;
- recovery of a prescribed sphere intrinsic width;
- full nonlamellar batch routing and output generation;
- the complete inherited lamellar V1 regression suite.

See `REFERENCES.md` for the method-to-literature map.
