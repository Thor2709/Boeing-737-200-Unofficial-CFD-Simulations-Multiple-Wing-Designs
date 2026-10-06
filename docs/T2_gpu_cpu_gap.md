# T2 GPU vs C5 CPU: why CL differs by 10.6 % (analysis 2026-10-05)

Inputs:
- CPU: `cfd_v2/runs/C5/iH0/final.{cas,dat}.h5`. Density-based implicit, double precision, 4 cores, 5,500 iterations.
- GPU: `cfd_v2/runs/C5/gpu_T2/autosave-1.cas.h5` + `autosave-1-05500.dat.h5`. Native GPU, pressure-based coupled, single precision, 5,900 iterations, timed out before the final write.

Method: offline h5py reading of the stored fields; no Fluent session.
- Wall pressure and wall-shear force per face; Newell area vectors from mesh nodes.
- Settings diff of the stored Rampant variables (23 of 9,591 differ).
- Transcripts and the history of both runs.
- Scripts: scratchpad `cmp_fields.py` and `cmp_figs.py`. Figures: `research/figs_t2_gap/`.

## 1. Ruled out (identical, or too small to matter)

| Check | Result |
|---|---|
| Mesh | Same file, same face/zone ordering. Wall-adjacent cell wall distance agrees to a median 0.0 % (p95 0.8 %). |
| Free stream | Same M 0.740, p, T, flow direction (α 3.624°). Operating pressure 0 in both. |
| Physics models | Same SST coefficients, Sutherland, ideal gas and far-field turbulence. |
| Force bookkeeping | My independent integration of the stored wall pressure + shear reproduces both solver totals: CPU CL 0.4882 vs reported 0.4882; GPU 0.5417 vs 0.5415. No report or monitor error. |
| Single-precision geometry | float32 coordinate resolution is 1–2 µm. The first cell is about 200 µm (y+ median 41), so the ratio is ≤ 0.01. Not a cause. |
| Convergence | CPU drifts −0.001 CL per 1,000 iterations. GPU is stationary from about iteration 500, with a ±0.0008 limit cycle. Both are converged states, and they differ. |
| GPU start-up | Temperature clipping (5,000 K cap) occurred only in the first ~40 iterations. One spurious cell (Mach 2.23, wing/pylon junction y 5.07 m) has no force effect. |

## 2. What is physically different: the extent of shock-induced separation

Both solutions have a strong upper-surface shock: peak isentropic Mach 1.4–1.6, above the ~1.3 at which shock-induced separation sets in. Behind that shock, the CPU separates much more.

| Reverse-flow wall area | CPU | GPU |
|---|---|---|
| Wing (almost all upper surface) | 7.23 m² (9.7 % of the wing) | 3.11 m² (4.2 %) |
| Horizontal tail (root trailing edge) | 0.50 m² | 0.08 m² |
| Fuselage (aft body) | 1.95 m² | 0.92 m² |
| Vertical tail | 0.05 m² | 0.03 m² |

Consequences:
- **Shock position.** With the larger separation bubble, the CPU shock sits further forward outboard: x/c 0.38–0.41 vs 0.41–0.46 (CPU vs GPU) at y 8.5–13 m. Inboard (y ≤ 5.5 m) the shocks coincide.
- **Lift.** CPU sectional lift is lower:
  - y 2.5–4 m: −3 %;
  - y 7–13 m: −9 to −10 %.
- **Where the lift goes.** The wing loses lift on both surfaces, upper ΔCL 0.019 and lower 0.011 (viscous decambering through the trailing-edge pressure).
- **Pattern.** The CPU separates more everywhere: wing shock foot, tail root, aft fuselage. This systematic pattern points to the numerical scheme, not to a single unstable bubble.

Figures:
- `figs_t2_gap/cp_stations.png`: Cp at 4 span stations.
- `figs_t2_gap/separation_dcp_maps.png`: reverse-flow maps and the ΔCp map. The ΔCp map is a thin band along the shock line.

## 3. Component bookkeeping (pressure part; half model, S 45.52 m²)

| Zone | CLp CPU | CLp GPU | ΔCL | CMp CPU | CMp GPU |
|---|---|---|---|---|---|
| Wing | 0.4141 | 0.4442 | +0.030 | −0.079 | −0.094 |
| Horizontal tail | 0.0097 | 0.0235 | +0.014 | −0.039 | −0.094 |
| Fuselage | 0.0612 | 0.0702 | +0.009 | +0.071 | +0.046 |
| Total | 0.4882 | 0.5417 | +0.054 | −0.066 | −0.162 |

**CD is equal only by coincidence of compensation:**
- Pressure drag: CPU 388 counts, GPU 376 counts.
- Friction drag: CPU 91 counts, GPU 101 counts. The attached flow has more skin friction.
- The GPU's 11 % more lift should cost about 25 counts of induced drag (AR 8.8, e ≈ 0.8). So at equal CL the GPU drag would be roughly 20–35 counts (≈5 %) lower. This is an estimate.

**CM is not fine.** It differs by a factor of 2.5, driven by:
- the aft shock (wing CM);
- the horizontal-tail lift (2.4×, because the CPU separates the tail root);
- the fuselage moment.

## 4. Numerical differences that can produce this (ranked)

1. **Discretisation family.**
   - CPU: density-based Roe-type upwind flux, implicit.
   - GPU: pressure-based coupled with Rhie-Chow momentum interpolation.
   - The two have different numerical dissipation in shocks and in the low-speed near-wall layer. Upwind density-based fluxes are known to be more dissipative at low local Mach numbers, which thickens the boundary layer and promotes separation. For Fluent's implementation this is a hypothesis, not yet verified.
2. **Gradients on the polyhedral mesh.**
   - The GPU solver switched warped-face gradient correction ON→OFF. The transcript lists it as an unsupported setting; this is a documented GPU limitation.
   - The gradient flags also differ: `recon/cell-lsf?` and the symmetry-plane gradient options (`nbg-hyb-skip-symmetry?`, `nbg-improved-sym-per?`).
   - These affect near-wall gradients and therefore the separation.
3. **Solution path (possible non-uniqueness).**
   - CPU: hybrid init at 3.40°, then a secant walk 3.40 → 3.50 → 3.55 → 3.43 → 3.62°, each step continued from the separated field.
   - GPU: cold far-field init straight at 3.62°.
   - Transonic RANS with shock-induced separation can have more than one steady solution and hysteresis near buffet onset.
   - The tail-root and aft-body differences argue for cause 1 or 2 rather than for path dependence.
4. **Precision (single vs double).** Ruled out for geometry; its only visible effect is the ±0.08 % limit cycle.

**Underlying reason for the sensitivity:** the C5 wing runs at α 3.6° on its lift break. The CPU wing lift slope there is about 0.05/deg, against about 0.10 attached, and the shocks are at Mach 1.4–1.6. In that regime small numerical differences shift the separation, and the separation shifts CL by about 10 %. A real 737-200 at cruise CL has much weaker shocks. This ties to the geometry accuracy being investigated (airfoil, twist).

## 5. Decisive tests (not yet run; each needs the licence free)

1. **CPU density-based restart from the GPU field** (autosave-1-05500, α 3.624, about 2–3 h).
   - Returns to CL ≈ 0.488: the scheme is the cause.
   - Stays near 0.54: two steady solutions exist (hysteresis), and the CPU result depends on its path.
2. **GPU restart from the CPU final field** (about 15–20 min). This is the mirror test.
3. **Isolation runs** (optional):
   - CPU with warped-face gradient correction off;
   - GPU in double precision.
4. **Grid sensitivity.** The chain's g400 and g650 CPU runs show the CPU's own grid sensitivity. g400 read CL 0.551 at iteration 300, which is far too early to interpret.

Note: plan step T5 points to `gpu_T2/final.dat.h5` and `--seed-from gpu_T2` (needs result.json). Neither exists because T2 timed out, so T5 fails closed and the chain goes to the CPU cold route. Test 1 needs a manual run with the autosave.

## 6. Implications

- The GPU route stays out. The early gate failed closed, and CL 10.6 % ≫ 1.5 %. The CPU remains authoritative.
- Reported CPU values need a caveat: in this flow regime the numerical-scheme uncertainty on CL is of order 10 % until the tests above and the GCI say otherwise.
- The GPU transient runs (pressure-based) will sit in a less-separated state than the CPU statics. Compare them only at matched conditions.

Sources:
- [Fluent GPU solver limitations](https://ansyshelp.ansys.com/public/Views/Secured/corp/v252/en/flu_ug/flu_ug_sec_gpu_solver_limitations.html)
- [Reading case files into the GPU solver](https://ansyshelp.ansys.com/public/Views/Secured/corp/v242/en/flu_ug/flu_ug_sec_gpu_solver_read_case.html)
- [Multiple solutions of transonic flow (NACA 0012)](https://archive.aps.org/dfd/2012/h24/1)
- [Multiple solutions and stability of transonic flows](https://www.csrc.sdsu.edu/?p=710)
