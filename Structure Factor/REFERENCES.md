# Method references

The S(q) pipeline uses these sources for its physical definitions and numerical
methods. They do not establish that a detected peak identifies a unique
morphology or proves thermodynamic equilibrium.

1. A. A. Gavrilov, Y. V. Kudryavtsev, and A. V. Chertovich, “Phase Diagrams
   of Block Copolymer Melts by Dissipative Particle Dynamics Simulations,”
   *J. Chem. Phys.* **139**, 224901 (2013),
   [doi:10.1063/1.4837215](https://doi.org/10.1063/1.4837215).
   The DPD block-copolymer precedent for calculating a static structure factor
   from particle-coordinate Fourier sums. The pipeline's `exact` engine follows
   that direct-sum definition, while sampling every reciprocal-lattice vector
   in its configured range, including mixed-sign vectors.

2. A. B. Bhatia and D. E. Thornton, “Structural Aspects of the Electrical
   Resistivity of Binary Alloys,” *Phys. Rev. B* **2**, 3004–3012 (1970),
   [doi:10.1103/PhysRevB.2.3004](https://doi.org/10.1103/PhysRevB.2.3004).
   Basis for separating number-density and concentration fluctuations in a
   binary system. The pipeline reports the Bhatia–Thornton-style concentration
   spectrum for the selected A/B pair:

   `S_cc = x_B² S_AA + x_A² S_BB − 2 x_A x_B S_AB`,

   where `x_A = N_A/(N_A + N_B)` and `x_B = N_B/(N_A + N_B)`.
   `S_cc_normalized = S_cc/(x_A x_B)` is an additional convention used here
   for comparisons across global A/B compositions.

3. R. W. Hockney and J. W. Eastwood, *Computer Simulation Using Particles*
   (Adam Hilger, 1988),
   [publisher edition](https://www.routledge.com/Computer-Simulation-Using-Particles/Eastwood-Hockney/p/book/9780852743928).
   Reference for cloud-in-cell particle assignment and particle-mesh Fourier
   methods. The pipeline's `cic` engine deposits beads on a periodic mesh,
   transforms the density with an FFT, and divides by the known CIC assignment
   window before radial averaging. Mesh results are checked against direct
   particle sums at low q because deconvolution does not remove all mesh error.

4. SciPy developers, [“`scipy.signal.find_peaks`”](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html)
   and [“`scipy.signal.peak_prominences`”](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_prominences.html),
   *SciPy API reference*.
   Documents the local-maximum and prominence concepts used when the optional
   SciPy peak finder is available. The pipeline's height, prominence, occupied
   bin, and search-range cutoffs are configurable analysis choices, not
   thresholds prescribed by these references.

The relation `L0 = 2π/q_star` converts a detected finite-q peak position to
its corresponding real-space period. The automatic averaging-window and
spectral-stationarity checks are pipeline heuristics; their thresholds should
be reported with each result. Stationarity over the sampled interval alone
does not distinguish an equilibrium state from a long-lived metastable state.

All spectra in this pipeline use only the selected `A_type_id` and `B_type_id`
and normalize by `N_A + N_B`. These binary definitions do not give a complete
multicomponent structure-factor analysis when other bead types are present.
