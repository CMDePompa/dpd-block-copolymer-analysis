# Method references

This interface analysis method uses these papers as method constraints, not as claims that every liquid or
homopolymer result transfers quantitatively to block-copolymer melts.

1. A. P. Willard and D. Chandler, “Instantaneous Liquid Interfaces,” *J. Phys.
   Chem. B* **114**, 1954–1958 (2010),
   [doi:10.1021/jp909219k](https://doi.org/10.1021/jp909219k).
   Basis for defining an instantaneous interface as a level set of a spatially
   coarse-grained field and measuring profiles relative to the nearest local
   interface and its normal, without assuming planar symmetry.

2. M. Muller and G. Munster, “Profile and Width of Rough Interfaces,” *J.
   Stat. Phys.* **118**, 669–686 (2005),
   [doi:10.1007/s10955-004-8824-2](https://doi.org/10.1007/s10955-004-8824-2).
   Basis for distinguishing intrinsic and capillary-wave broadening and for
   reporting the second moment of the profile gradient as a width.

3. S. W. Sides, G. S. Grest, and M.-D. Lacasse, “Capillary Waves at
   Liquid/Vapor Interfaces: A Molecular Dynamics Simulation,” *Phys. Rev. E*
   **60**, 6708–6713 (1999),
   [doi:10.1103/PhysRevE.60.6708](https://doi.org/10.1103/PhysRevE.60.6708).
   Supports derivative-moment widths, the additive variance interpretation of
   convolution broadening, and retaining an error-function estimator because
   tanh fits can bias second moments.

4. M.-D. Lacasse, G. S. Grest, and A. J. Levine, “Capillary-Wave and
   Chain-Length Effects at Polymer/Polymer Interfaces,” *Phys. Rev. Lett.*
   **80**, 309–312 (1998),
   [doi:10.1103/PhysRevLett.80.309](https://doi.org/10.1103/PhysRevLett.80.309).
   Polymer-interface basis for separating intrinsic width from capillary-wave
   broadening and for treating width through profile-derivative moments.

5. A. Werner, F. Schmid, M. Muller, and K. Binder, “Intrinsic Profiles and
   Capillary Waves at Homopolymer Interfaces: A Monte Carlo Study,” *Phys.
   Rev. E* **59**, 728–738 (1999),
   [doi:10.1103/PhysRevE.59.728](https://doi.org/10.1103/PhysRevE.59.728).
   Supports centering local profiles on the instantaneous interface and makes
   explicit that the inferred intrinsic profile depends on coarse-graining
   scale.

6. M. Sega et al., [PyTIM documentation](https://marcello-sega.github.io/pytim/).
   Implementation precedent for Willard–Chandler-style mesh surfaces and for
   nonplanar interfacial analysis. V2 does not depend on PyTIM.

7. H. Jinnai et al., “Direct Measurement of Interfacial Curvature
   Distributions in a Bicontinuous Block Copolymer Morphology,” *Phys. Rev.
   Lett.* **84**, 518–521 (2000),
   [doi:10.1103/PhysRevLett.84.518](https://doi.org/10.1103/PhysRevLett.84.518).
   Establishes mean- and Gaussian-curvature distributions as meaningful
   descriptors of bicontinuous block-copolymer interfaces.

8. I. Prasad, H. Jinnai, R.-M. Ho, E. L. Thomas, and G. M. Grason, “Anatomy
   of Triply-Periodic Network Assemblies: Characterizing Skeletal and
   Inter-Domain Surface Geometry of Block Copolymer Gyroids,” *Soft Matter*
   **14**, 3612–3623 (2018),
   [doi:10.1039/C8SM00078F](https://doi.org/10.1039/C8SM00078F).
   Used as contextual support for local interfacial curvature and skeletal
   network descriptors in complex block-copolymer domains.

The curvature formulas themselves are standard implicit-level-set identities:
with `g = grad(phi)` and Hessian `D2(phi)`, V2 evaluates

`H = (|g|^2 tr(D2phi) - g^T D2phi g) / (2 |g|^3)`

and

`K = g^T cof(D2phi) g / |g|^4`.

All derivative quantities are computed from the smoothed periodic composition
field and area-weighted on the extracted interface.
