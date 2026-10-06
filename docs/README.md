# Documentation Directory Index

This directory contains technical notes, methodology references, and CFD investigation reports for the Boeing 737-200 aerodynamic evaluation:

- [README.md](README.md): Overview and table of contents for all project technical notes and documentation.
- [METHODS.md](METHODS.md): Numerical methodology covering CAD generation, aerofoil repanelling, flow5 panel methods, poly-hexcore meshing, steady/transient CFD, and post-processing.
- [lessons_learned.md](lessons_learned.md): Chronological technical log of solver behaviors, numerical stiffness, GPU/CPU constraints, meshing findings, and pipeline operations.
- [geometry_discrepancies_737.md](geometry_discrepancies_737.md): Aerodynamic and geometric discrepancy audit comparing the CAD model with real Boeing 737-200 flight and polar data.
- [T2_gpu_cpu_gap.md](T2_gpu_cpu_gap.md): Investigation into the 10.6% lift gap between density-based CPU and pressure-based native GPU solvers at transonic buffet conditions.
- [drift_analysis.md](drift_analysis.md): Analysis of slow iterative convergence modes in the aft body and tail-root junction under density-based time-marching.
- [alpha_seed_method.md](alpha_seed_method.md): Method and formulas for estimating CFD starting angles of attack from 2D sweep-normal aerofoil polars and flow5 data.
- [validation_report.md](validation_report.md): Verification and validation report assessing grid sensitivity, wall y+ resolution, component loads, and nacelle closure drag.
