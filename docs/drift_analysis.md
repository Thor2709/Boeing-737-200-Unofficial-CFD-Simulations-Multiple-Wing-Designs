# C5 slow CL/CM drift: analysis (2026-10-04 10:40)

Data: `cfd_v2/runs/C5/iH0/history.csv` (per-zone forces every 50 it), transcript residuals. Run: CPU density-based implicit, SST, CFL 10, 997,858 cells.

## 1. What drifts (per-zone, alpha 3.554, it 2300-3550, 3-point exponential fit)
| Quantity | Last | Asymptote | Time constant (it) | Moves? |
|---|---|---|---|---|
| CL wing | 0.41086 | 0.41084 | 243 | settled within ~500 it of each alpha step |
| CL htail | 0.00954 | 0.00916 | 539 | slow, falling (upload shrinking) |
| CM htail | -0.0381 | -0.0367 | 538 | slow |
| CL / CM / CD fuselage | 0.0601 / +0.0702 / 119.6 cts | 0.0599 / +0.0709 / 120.0 | 560-610 | slow, CD rising |
| Nacelle, duct, fairing, vtail | — | — | — | flat |
| **Total CL / CM** | 0.48355 / -0.0659 | **0.48311 / -0.0638** | 420 / 525 | slow |
Over the whole run at alpha 3.4: htail CL 0.0208 -> 0.0141, CM_htail -0.083 -> -0.056, CM_fuselage +0.046 -> +0.060, fuselage CD 143 -> 114 cts while the wing settled by it ~1000.
Residuals at it 3555: continuity 2.6e-5, momentum 3-6e-5, energy 2.5e-5, k 2.7e-5, omega 9.3e-5, still falling; mass imbalance flat 0.0043 (far-field face-value floor, separate issue).

**Conclusion:** the wing converges fast. The slow mode lives in the aft fuselage + horizontal tail (the region fed by the wing wake/downwash and the tail-cone flow). It is a monotone exponential approach (no oscillation), so it is slow iterative convergence of a steady solution, not unsteadiness.

## 2. Causes (ranked)
1. **Density-based solver stiffness in low-speed regions (likely main cause).** The aft body / tail-cone region has low local Mach (thick boundary layer, possible small separation behind the tail-cone cap). Density-based time-marching schemes converge slowly where Mach is low unless low-Mach preconditioning is used (Weiss-Smith; Storti 1992; literature speed-ups up to ~20-28x). Fluent's density-based solver at M 0.74 freestream does not precondition. Fits: the slow zones are exactly the low-speed ones; the transonic wing is fast.
2. **Slow growth of a small aft-body separation / tail-root flow.** Fuselage CD rises while tail upload falls: consistent with a thickening aft-body boundary layer changing the downwash at the tail. RANS needs many iterations to settle separation size (DPW-IV: lift and pitching-moment scatter came from side-of-body separation size). Here the wing (and the fairing zone) are flat, so it is the aft body, not the wing root.
3. **Pseudo-time limit (CFL 10).** A global slow mode scales with CFL; raising CFL shortens it (the Courant A/B on g400 tests this directly).
4. Geometry checks to do in post-processing: tail-cone planar cap (base flow), vertical-tail root crossing the symmetry plane by ~0.2 m (G1 recon of the source geometry); look at wall shear/separation lines on the aft body in the C5 post images.

## 3. Is it an error?
- **The physics/answer: no.** Residuals keep falling, the approach is monotone, and the asymptote is well defined (CL 0.4831, CM -0.064 at alpha 3.554). The converged state is a valid steady RANS solution.
- **The search logic: yes.** The drift rule (|dCL| < 0.001 per 100 it) passes when tau ~ 550 it still leaves ~0.003-0.005 of CL to go. Result: secant points taken before settling (alpha 3.501 after 450 it) gave a negative slope and a wrong-direction step (3.554 -> ~3.44). Self-corrects in ~25 min, costs ~2 h total.
- **For the results:** CL/CD are fine once settled (CD flat already). **CM is the sensitive one** (moved 0.09 over the run); any CM-based use (trim, the optional C5 trimmed run, tail-load interpretation) must take CM from fully settled data.

## 4. Fixes
- **Search:** settle test by exponential extrapolation (|y_last - y_asym| < tol for CL and CM, using a 3-point fit over the last ~600 it) instead of a fixed per-100-it drift; reject secant slopes outside 0.05-0.15 /deg (use the seed slope); re-step with the known slope if the final run drifts out of tolerance.
- **Run strategy:** GPU pressure-based coupled has no low-Mach stiffness, so T2 should settle the aft body much faster; the warm start then hands the CPU a settled tail flow. Courant A/B shows how much CFL 20 shortens tau.
- **Report:** state tau per zone and the extrapolated remaining change (iterative-convergence uncertainty) next to GCI.

Sources: Weiss & Smith preconditioning / Storti 1992 (https://scipedia.com/public/Storti_et_al_1992a); NASA NTRS 19930005696; Giles (https://people.maths.ox.ac.uk/gilesm/files/eccomas01b.pdf); DPW-IV summary (https://ntrs.nasa.gov/api/citations/20100026466/downloads/20100026466.pdf); Ansys forum transonic convergence (https://innovationspace.ansys.com/forum/forums/topic/convergence-issue-for-transonic-flow).
