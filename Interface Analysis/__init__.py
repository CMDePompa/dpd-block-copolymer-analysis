"""Morphology-aware interface analysis for block-copolymer DPD melts.

A field-based interface characterization pipeline for A/B block-copolymer
LAMMPS/DPD trajectories. It coexists with
(and reuses conventions from) the ``Sq_pipeline_v4`` package in this project
but never modifies it or its outputs.

V2 routes by the user-declared morphology. Lamellae retain the validated V1
height-field analysis. Spheres, cylinders, and gyroids use a fully periodic
three-dimensional phi=0 isosurface, local-normal intrinsic profiles,
curvature, connected-domain geometry, and topology. It deliberately does not
classify morphology.

Shared physical foundation (per frame):
  1. Deposit A and B bead positions onto a periodic grid (cloud-in-cell) and
     smooth with a periodic Gaussian to obtain rho_A(r), rho_B(r).
  2. Composition field  phi = (rho_A - rho_B) / (rho_A + rho_B); the nominal
     A/B interface is the level set phi = 0.
  3. Route to lamellar height fields or a general periodic 3-D surface.

Units: DPD reduced units throughout (r_c = 1, bead number density rho = 3).
Notation: q = wavevector (2*pi/wavelength), L0 = 2*pi/q_star.
"""

from __future__ import annotations

__version__ = "2.0.0"

__all__ = ["__version__"]
