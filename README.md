# DPD Block Copolymer Analysis

Structure-factor and interface analysis workflows for LAMMPS/DPD block-copolymer trajectories. This is the first standalone release of the analysis collection (v1.0.0). The workflows retain their internal method labels, S(q) pipeline v4 and interface analysis V2.

## Workflows

- [Structure Factor](Structure%20Factor/Sq_README.md): computes A/B partial and composition S(q) using a fast particle-mesh calculation or an exact direct sum. It reports peak detection, characteristic period, numerical checks, and spectral stationarity. See its [method references](Structure%20Factor/REFERENCES.md).
- [Interface Analysis](Interface%20Analysis/Int_README.md): measures A/B interface geometry and mixing for user-declared lamellar, spherical, cylindrical, or gyroid morphologies. It reports local-normal width, curvature, domain geometry, and topology and can read S(q) results for a length scale and analysis window.

Both workflows operate on a **selected A/B pair**, set with `A_type_id` and `B_type_id`. They can read trajectories containing other bead types, but do not provide a complete multicomponent analysis. S(q) excludes additional types from its spectra and normalizes by `N_A + N_B`. Interface density fields exclude them, while some bead-level mixing fractions count all particles; optional chain-region profiles reject extra types. Interpret A/B results from a system with more than two species accordingly.

## Quick start

Use Python 3.10 or newer. Install each workflow's dependencies in the environment used to run it:

```bash
python -m pip install -r "Structure Factor/requirements.txt"
python -m pip install -r "Interface Analysis/requirements.txt"
```

From `Structure Factor/`:

```bash
python run_pipeline.py --make-template config.yaml
python run_pipeline.py --config config.yaml --test-mode
python run_pipeline.py --config config.yaml
```

From `Interface Analysis/`:

```bash
python run_interface.py --make-template config.yaml
python run_interface.py --config config.yaml --test-mode
python run_interface.py --config config.yaml
```

Edit the generated templates for your trajectory and output paths. Read the workflow guides for equations, configuration choices, interpretation, and limitations. The interface workflow uses a morphology you declare after inspecting the data; it does not classify morphology automatically.

The input-configuration builder is maintained in [DPD-for-Block-Copolymers](https://github.com/CMDePompa/DPD-for-Block-Copolymers).

License: GPL-3.0-or-later. See `LICENSE` and `citation.cff`.
