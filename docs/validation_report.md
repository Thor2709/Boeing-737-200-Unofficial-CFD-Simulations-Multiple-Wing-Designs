# B737-200 CFD validation report (Fluent 26.1, M 0.74, 30,000 ft, CL ≈ 0.489)

Date: 2026-10-03. Cases: medium meshes (275k cells) for `naca0009`, `opt2209`, `baseline`; NACA 0009 also on the fine mesh (454k).
Method: verification first, then validation, following AIAA G-077, ASME V&V20 and Roache GCI. The plan and log are in `validation_plan.md`; scripts, data and plots are in `validation/`.
No existing data was changed. New Fluent runs only read cases or wrote files under new names.

## 0. Summary verdict

| # | Check | naca0009 | opt2209 | baseline | Note |
|---|---|---|---|---|---|
| 1 | Iterative convergence | CAUTION | CAUTION | CAUTION | Residuals stall at 1.5e-3 / 5e-3 / 1e-2. Forces are steady to < 1 drag count per 100 iterations. |
| 2 | Mass and far-field | PASS | PASS | PASS | Imbalance 3e-5 of the throughflow; far-field Cp within ±0.002 and M 0.738–0.740. |
| 2 | Domain size | CAUTION | CAUTION | CAUTION | 31 MAC upstream and 29 MAC to the sides is acceptable; no far-field vortex correction. |
| 3 | Wall treatment (y+) | FAIL | CAUTION | FAIL | Median y+ 16–21 (buffer layer, not 50). Prisms collapse on the aft 20–30% chord of the lifting surfaces, where y+ reaches 10³–10⁴. |
| 4 | Component loads | PASS (method) | PASS | PASS | Fluent splits are exact and sum to the reported CL/CD. |
| 4 | Drag level against the 737 reference | FAIL | CAUTION | FAIL | The closed blunt nacelle adds about 0.013–0.015. With that removed: 0.042 / 0.034 / 0.040, against 0.04315. |
| 4 | Trim / tail load | FAIL | FAIL | FAIL | Untrimmed: CM −0.22 / −0.23 / −0.19 about x = 12.44 m. The tail carries upload, not download. |
| 5 | Cp physics (stagnation, shocks, trailing edge) | PASS (qualitative) | PASS | PASS | Shocks are strong (M_pre 1.3–1.5) and smeared over 2–3 cells. |
| 6 | Mesh sensitivity | CAUTION | n/a | n/a | Fine vs medium: CL +1.6%, CD −1.9%. Two grids only, GCI 12–14%. The coarse mesh was never solved. |
| 7 | Cross-case consistency | PASS | PASS | PASS | Δα0 and CL-α agree with thin-aerofoil theory to 80–90% (the wing carries ~85% of CL). |
| 9 | Overall | **Caution: usable for ranking** | **Caution: usable for ranking** | **Caution** | Absolute CD is not credible until the nacelle and trailing-edge mesh are fixed. |

**Bottom line.** The CL and α results and the case ranking (opt2209 < baseline < naca0009 in drag) are credible.
The absolute drag is not. About 0.013–0.015 of every case's CD (25–30%) comes from modelling the nacelle as a closed blunt body.
Of the 98-count drag gap between naca0009 and opt2209, only about half is on the wing itself. The rest is lower-α interference on the tail, nacelle and fuselage. The ranking holds; the size of the gap is mesh-limited.

## 1. Convergence

Plot: `validation/conv_history.png`, `drift.png`; data `data/conv_stats.json`, `data/drift.json`.
- **Residuals at stop (scaled):**
  - naca0009: continuity 1.5e-3
  - opt2209: 4.8e-3
  - baseline: about 1e-2
  - In all three they are flat and stalled, not falling. k and ω are 2e-4 and 5e-5.
  - No "turbulent viscosity limited" messages on the medium meshes; the fine mesh had about 380 cells limited.
- **The original pipeline stopped too early.**
  - `settle()` accepted a 0.2% change between 50-iteration chunks.
  - The final-α segments had only 200–400 iterations.
  - At α 3.0, CD was still falling 4 counts per 100 iterations, about −3% over the last 400.
- **Fixed-α continuation (`solve2.py`, residual stop disabled):**
  - naca0009 medium, 375 further iterations: CL 0.4916 → 0.4921, CD 0.05697 → 0.05663 (−3.4 counts, −0.6%), CM −0.2200 → −0.2197.
  - Over the last 200 iterations the drift is CL +7e-5 and CD −0.9 count per 100 iterations; most of it is in the fuselage zone.
  - opt2209, 250 iterations: CL 0.4885 → 0.4873, CD 0.04721 → 0.04688 (−3.3 counts), still −1.1 count per 100 iterations.
  - baseline, 250 iterations: CL 0.4887 (unchanged), CD 0.05437 → 0.05415 (−2.2 counts), −0.75 count per 100 iterations.
  - All three move in the same direction by similar amounts, so the case ranking is unaffected.
- **Verdict:** iterative error in CD ≈ 0.5–1% (3–6 counts). That is small against the case differences (70–100 counts) but not negligible against the target of ~1 count. CAUTION.

## 2. Conservation and boundary checks

- **Net mass flux through the far field:** +67, −71 and −78 kg/s against a throughflow of 2.2e6 kg/s, an imbalance of 3e-5. The Fluent flux report for naca0009 gives −1.33 kg/s. PASS.
- **Far-field state:** Cp −0.002…0.000 and M 0.738–0.740 on the far-field faces, so the boundary is undisturbed. PASS.
- **Domain:** x −100…170 m, y 0…100 m, z ±100 m, which is 31 MAC upstream, 47 MAC downstream and 29 MAC to the sides.
  - This is adequate for a pressure far field at M 0.74, but there is no point-vortex correction.
  - A residual induced-angle error of ~0.05° is expected, which would reduce CL by about 0.006 at fixed α. CAUTION.
- **Symmetry plane:** maximum M 0.99–1.01 (fuselage crown), maximum Cp 1.12 (nose stagnation, isentropic 1.144). No anomalies.
- **Mesh quality:** minimum orthogonal quality 7.3e-3 at (27.2, 0.11, 4.03), at the fin–fuselage junction. Maximum aspect ratio 4421 (prisms).
- **Fluent Meshing reported faces with "invalid dihedral angle":** 192 (naca0009), 486 (opt2209) and 194 (baseline). See §8.

## 3. Wall treatment (y+, Cf)

Plot: `validation/yplus.png`. Medium meshes: first layer 2e-4 m, 12 layers, growth 1.25.

| Component | median y+ | share of area with y+ > 300 (naca / opt / base) |
|---|---|---|
| Fuselage | 16 | 0% |
| Outboard wing | 18–21 | 37% / 7% / 33% |
| Inboard wing | 19 | 13% / 6% / 12% |
| Horizontal tail | 20 | 26% / 25% / 27% |
| Vertical tail | 19 | 8–10% |
| Nacelle | 12–13 | 5% |

- **y+ is in the buffer layer, not ~50.** At y+ ~ 15–20 the SST wall treatment blends, so skin friction is typically within ±5%. The results summary should not say "y+ ~ 50".
- **Prisms collapse on the aft part of the lifting surfaces.**
  - Over the aft 20–30% of chord on the wing (naca0009, baseline) and both tails, the first cell is a core polyhedron with y+ 10³–10⁴.
  - In these regions friction is under-predicted and trailing-edge pressure recovery is wrong.
  - opt2209 kept its prisms on most of the wing (7%), which flatters its comparison slightly.
  - Viscous wing CD is nevertheless almost equal: 0.00315 / 0.00314 / 0.00346.
- **Friction level:** fuselage CDv 0.0049–0.0053 against a flat-plate estimate of ~0.0050 (component build-up); plausible. Mid-span Cf where prisms exist is 0.0015–0.002. FAIL for wing and tails, PASS for the fuselage.

## 4. Component loads (exact Fluent integration)

Method: each wall zone is split in session by a 35° feature angle (`split.py`), Fluent `wall_forces` reports each piece in the wind axes, and pieces are classified by geometry (`components.py`, data `data/components.json`).
- **The `mainwing` zone is not shrunken.** It holds only the outboard panels (y > 4.70 m, 51.1 m²).
  - The inboard wing panel (24.3 m²) and the inboard lower surface of the tailplane (3.5 m²) were meshed into the `fuselage` zone, which is why that zone carries CL 0.17.
  - The total wetted wing area is 75.0–75.6 m² per side, which matches the VSP planform: exposed area about 37 m² per side, times about 2.03.

| CL / CD (×1e-4) / CDp / CDv | naca0009 α 3.65° | opt2209 α 2.13° | baseline α 2.62° |
|---|---|---|---|
| Wing (outboard, inboard, tip) | 0.394 / 242 / 211 / 32 | 0.418 / 194 / 163 / 31 | 0.403 / 236 / 202 / 35 |
| Fuselage | 0.064 / 146 / 94 / 53 | 0.058 / 133 / 84 / 50 | 0.062 / 142 / 92 / 49 |
| Horizontal tail | 0.029 / 21 / 10 / 12 | 0.012 / 4 / −8 / 12 | 0.016 / 8 / −4 / 11 |
| Vertical tail | 0.004 / 7 | 0.004 / 7 | 0.004 / 7 |
| Nacelle inlet disc (closed, flat) | −0.001 / **184** | −0.001 / **185** | −0.001 / **184** |
| Nacelle body and pylon | 0.002 / −36 | −0.002 / −52 | 0.004 / −37 |
| Nacelle base | 0 / 5 | 0 / 1 | 0 / 4 |
| **Total** | **0.4916 / 570 / 464 / 105** | **0.4885 / 472 / 370 / 102** | **0.4886 / 544 / 438 / 105** |
| CM about (12.44, 0, 0), Lref 3.21 | −0.220 | −0.226 | −0.194 |
| L/D | 8.6 | 10.4 | 9.0 |

- **Nacelle.**
  - The VSP nacelle is a closed body with a flat front disc 1.07 m in diameter, at stagnation pressure (184 counts on its own).
  - The forebody suction recovers 36–52 counts, so the net engine-zone drag is 133–153 counts.
  - A flow-through nacelle would give about 15 counts (build-up), so **the closure adds about 120–140 counts to every case**. This is the largest single error in the absolute CD.
- **Against the references** (`data/reference_values.json`; Excel Main Sheet Q2/W2: real 737-200 CD 0.04315, L/D 11.33; the brief's "0.044" is close to this):

  | | naca0009 | opt2209 | baseline |
  |---|---|---|---|
  | CFD CD | 0.0570 | 0.0472 | 0.0544 |
  | CFD CD less the nacelle-closure increment | 0.042–0.044 | 0.034–0.036 | 0.040–0.042 |
  | L/D after that correction | 11.2–11.7 | 13.6–14.4 | 11.7–12.2 |

  - Both naca0009 and baseline are then within ±5% of the real aircraft.
  - The flat-plate/form-factor build-up (`sec_drag.md`) gives 0.031–0.033 for the flow-through configuration. That is a lower bound: no excrescences, trim or real-wing wave drag.
- **Wing 2-D CD ~0.0045 (Excel Q5: NACA 0009 = 0.00449):**
  - Wing friction plus form in CFD is about 0.0032 (viscous) plus 0.002–0.004 (pressure form). This is consistent.
  - The remaining wing CDp after removing CDi_wing ≈ CL_w²/(π·AR·0.8) ≈ 0.0071–0.0080 is 0.014 (naca0009), 0.008 (opt2209) and 0.013 (baseline). This is wave plus form drag; the shocks in §5 explain why it is high.
- **CL-α:** 0.124 (naca0009), 0.125 (baseline), 0.115 (opt2209) per degree, from the α-search points. The DATCOM-type estimate for the complete aircraft at M 0.74 is about 0.12/deg. PASS.
- **Tail load:**
  - The horizontal tail carries an upload of CL +0.012 to +0.029; there is no download.
  - With 0° tail incidence and CM −0.19 to −0.23 about the 25% MAC point, the aircraft is far from trim.
  - Trimming would need a tail download of roughly ΔCL_t ≈ −0.07 (CM/ l_t: 0.22·3.21/14 m). That costs about +0.07 wing CL at the same total CL, plus a few counts of trim drag.
  - The measured downwash at the tail (x = 22.5 m, 14 faces) is 4.4–4.8°. It is indicative only.
  - FAIL: trim is outside the scope of the current runs, but it biases the CL split.

## 5. Surface pressure

Plots: `validation/cp_stations.png` (η = 0.10–0.90, theory overlay), `span_loading.png`, `cp_body_tails.png`.
The theory overlay is a 2-D linear-vortex panel method with Karman-Tsien correction and simple sweep (Λ¼ = 25°, Mn 0.671) at the CFD section cl. Cp* = −0.63, or −0.75 with the sweep adjustment.

| η | naca0009: cl, shock x/c, M before shock | opt2209 | baseline |
|---|---|---|---|
| 0.15 | 0.46, 0.40, 1.12 | 0.45, 0.47, 1.12 | 0.46, 0.43, 1.21 |
| 0.30 | 0.42, 0.20, 1.53 | 0.44, 0.33, 1.34 | 0.42, 0.34, 1.54 |
| 0.50 | 0.50, 0.27, 1.42 | 0.53, 0.33, 1.36 | 0.49, 0.32, 1.44 |
| 0.70 | 0.58, 0.22, 1.46 | 0.61, 0.32, 1.38 | 0.60, 0.32, 1.34 |
| 0.90 | 0.53, 0.22, 1.53 | 0.59, 0.38, 1.39 | 0.57, 0.35, 1.34 |

- **Shocks.** Every station from η 0.15 outboard carries a strong upper-surface shock.
  - NACA 0009 has a flat leading-edge suction plateau (Cp ≈ −1.5 to −1.66) ending in the most forward and strongest shock (x/c 0.20–0.27, M 1.42–1.53).
  - opt2209 accelerates gradually and has a weaker, more aft shock (M 1.34–1.39). This is consistent with its lower wing CDp (163 vs 211 counts).
  - M_pre > 1.3 means shock-induced separation is likely in reality. SST on this mesh shows trailing-edge Cp recovering to +0.07…+0.15, so there is no massive separation.
  - The shocks are smeared over 2–3 surface cells (surface mesh ~20–30 points per chord side). This is too coarse for shock position to ±2% chord.
- **Against theory.**
  - The subsonic panel method cannot represent the supersonic pocket, so it predicts a sharp leading-edge spike instead of the plateau. This is expected.
  - The lower surface and the region aft of the shock agree to ±0.1 in Cp.
  - The CFD lower surface of the baseline is slightly more negative at η 0.30–0.50, consistent with its thicker section.
- **Stagnation and trailing edge.**
  - The sampled stagnation Cp is 0.82–1.16 against an isentropic 1.144. Values below 1 occur where the cut misses the stagnation face on the coarse leading-edge surface mesh; there is no physical loss.
  - Trailing-edge Cp is +0.02 to +0.15 with closed upper/lower curves. PASS.
- **Junction and inboard (η 0.10 is inside the fuselage).**
  - The wing–body junction has no fairing; the η 0.15 section is strongly loaded (cl 0.46, cdp 0.047–0.081).
  - The baseline's thick root (t/c 15%) carries a long supersonic region (x/c 0.05–0.52), which is the source of its extra 29 counts over opt2209 on the inboard wing.
- **Spanwise loading** (`span_loading.png`): c·cl peaks inboard (y 2–3 m) and falls toward the tip, with no tip stall. Sectional cl rises outboard to 0.58–0.61 at η 0.7, which is outboard-heavy, as expected for 25° sweep with no washout (+1° twist on all sections).
- **Fuselage centreline:** nose Cp 1.12, crown suction over the wing, and a smooth tailcone recovery to Cp +0.2…+0.3 at x 28–29.5 m. PASS.
- **Nacelle:** stagnation over the whole front disc (§4), forebody suction down to Cp −1.3, base Cp ≈ 0. This is non-physical for a turbofan; the inlet should be flow-through.
- **Tails:** the horizontal-tail section Cp is mildly loaded upward; the fin is symmetric to ±0.05 in Cp (no sideslip). PASS.
- **Wake and downwash:**
  - The x = 35 m plane is too coarse (cell size ~1 m) for a Trefftz-plane estimate. CDi_Trefftz ≈ 0.001 and the total-pressure-loss CD of 0.12 are both numerical-dissipation artefacts and are **not used**.
  - The downwash at the tail is 4.4–4.8° (from 14 faces at y 2.3–4.3 m). The 1/(πAR)·2CL estimate is about 3.6°. Plausible, but low confidence.

## 6. Mesh sensitivity

Data and plots: `sec_mesh.md`, `mesh_sensitivity.png`, `data/mesh_gci.json`.
- **Meshes:** coarse 155k, medium 275k, fine 454k cells (r = 1.18 between fine and medium).
  - The coarse mesh was generated but never written because of a path error (spaces in the path), so it was never solved.
  - A fresh coarse/fine solve was dropped to keep within the 1-hour limit.
- **Why the fine α search diverged.**
  - At iteration 1095 the fine run met Fluent's default 1e-3 residual criterion (`runs/naca0009_fine_solve.log`, line 1602).
  - From then on every `iterate(50)` ran a single iteration, so new α values were never actually solved; only the force vectors were rotated.
  - At the step to α 3.56°, CD fell by 9 counts in one iteration (exactly CL·Δα), and CL did not change.
  - The secant slope came out ≈ 0, so the search jumped to 6.33° and then about 16°, where CL reached 1.0–1.4. Those later points are invalid.
  - The medium runs escaped this only because their residuals stayed above 1e-3.
  - Fix: disable the residual criterion (done in `validation/solve2.py`).
- **The quoted "CL +0.009, CD −4–5%" is partly an artefact.** It compares the rotated pseudo-point at α 3.56° with the medium result.
- **Valid comparison at α 3.65°:**

  | | medium (continued) | fine | difference |
  |---|---|---|---|
  | CL | 0.4918 | 0.4998 | +0.008 (+1.6%) |
  | CD | 0.05670 | 0.05565 | −10.5 counts (−1.85%) |

  - The fine solution was itself still drifting (CL −0.001 per 100 iterations).
- **GCI** (two grids, p = 2 assumed, Fs = 3): CL 12%, CD 14%. Richardson-extrapolated values: CL ≈ 0.52 and CD ≈ 0.053.
  - These are not tight bounds, but they show the medium mesh is not mesh-converged for the shocks: the drag differences between cases are larger than 1.85% but within the GCI band.
  - The trend (finer mesh → more lift, less drag) is typical of shock smearing and trailing-edge under-resolution.
- **Verdict:** CAUTION. Three solved grids at fixed α are needed for a real GCI.

## 7. Cross-case consistency

- **Zero-lift shift.** Thin-aerofoil and panel α0: NACA 0009 0°, opt2209 −1.85° / −1.96°, baseline −1.1° to −1.3° (`data/theory.json`).
  - CFD α at CL 0.489: Δα(opt2209 − naca0009) = −1.52° against a theoretical −1.9° (80%); Δα(baseline − naca0009) = −1.04° against −1.2° (87%).
  - These ratios are consistent with the wing producing about 82% of the lift (0.39–0.42 out of 0.49) and with fuselage, tail and nacelle lift that changes with α, not with camber. PASS.
- **Credibility of the 17% drag drop (opt2209 vs naca0009, −98 counts):**
  - Wing −48 counts (CDp −48, CDv 0): weaker, more aft shock. Physically consistent with the Cp.
  - Tail −17 counts and fuselage −13 counts: lower α (2.13° vs 3.65°) means less tail upload and less fuselage pressure drag.
  - Nacelle −20 counts: the closed nacelle at lower α picks up more forebody suction. This term is an artefact of the closed nacelle, and the same caveat applies to the baseline comparison.
  - **Verdict:** the ranking is credible. The wing-attributable gain is about −48 counts (−8.5%). Treat the full −17% as an upper estimate; with the mesh GCI and the nacelle artefact, ±20 counts is a fair band.
- **Baseline vs opt2209 (−72 counts):** wing −42, nacelle −15, fuselage −9, tail −4. Most of it is the thick inboard root shock of the baseline (η 0.15 cdp 0.081 vs 0.047).

## 8. Open issues

| Issue | Finding | Impact |
|---|---|---|
| L/R asymmetry | The half model uses the +y side for every case. Discovery volumes differ by < 1% between the nacelles (6.326 / 6.264 m³) and by 0.03% between the tailplanes. VSP itself is exactly symmetric. | Negligible (< 1 count) |
| "Shrunken" mainwing area | Not shrunken. The inboard panel (24.3 m²) is meshed in the `fuselage` zone. Total wing wetted area is 75 m² per side; the reference area 45.52 m² per half is correct. | Per-zone reports only; totals are correct |
| opt2209 Check Geometry issue | Discovery `inspect_geometry` reported 0 issues for the united models (`validation_summary.md`). Fluent Meshing found 486 faces with invalid dihedral angle for opt2209 against about 193 for the others, probably at the thin, cusped trailing edge. The opt2209 solution is no less smooth, and it kept more of its trailing-edge prisms. | Low; inspect the trailing-edge closure |
| No wing–body fairing | The junction is bare. The fuselage-zone pressure drag (84–94 counts) and the η 0.15 shock include junction effects. A real 737 has a belly fairing. The "fairing" bodies in VSP are the pylons (present, engine.5/6). | +5…15 counts estimated |
| y+ 50 vs resolved boundary layer | Actual median y+ is 16–21; prisms are lost on the aft chord. y+ ≈ 1 does not fit the student cell limit. Fix the prism collapse first (layer-stair-stepping at the trailing edge). | Friction ±5–10%; trailing-edge pressure recovery |
| Closed nacelle | Flat-disc inlet at stagnation. | +120…140 counts in every case |
| Untrimmed, tail at 0° incidence | CM −0.19…−0.23 | CL split between wing and tail; trim drag |

## 9. Verdict and prioritised fixes

- **naca0009:** CAUTION (medium confidence for CL and α; low for absolute CD). Valid as the reference case for ranking.
- **opt2209:** CAUTION (medium confidence). The best case; its drag gain over naca0009 is real, but about −50 counts on the wing rather than the −98 total.
- **baseline:** CAUTION (medium-low confidence). The thick root produces the strongest inboard shock, which the medium mesh resolves poorly.

Fixes, in priority order:
1. **Make the nacelle flow-through** (open inlet and exhaust, or an inlet/outlet boundary with mass flow), or report CD with the closed-nacelle increment (engine zone 133–153 counts, less about 15) subtracted. This is the largest error, at 25% of CD.
2. **Fix the trailing-edge prism collapse** on the wing and tails (prism stair-stepping, a blunt trailing-edge base, or a smaller first layer with fewer layers). Report the actual y+ ≈ 20.
3. **Iterative convergence:** disable the residual stop, run at least 1,000 iterations per α, and judge by force drift below 0.5 count per 100 iterations. Retarget CL with an α search that verifies each new α has been iterated.
4. **Mesh study:** write and solve the coarse mesh (quote the path) and rerun the fine mesh at fixed α. Run a three-grid GCI on CL, CD and CM. Refine the surface around the shocks (x/c 0.15–0.5) so they are resolved to about 1% chord.
5. **Trim:** set the tailplane incidence (or add a trim iteration) so that CM ≈ 0 about the chosen CG, and compare the cases at trimmed CL.
6. **Add a wing–body fairing** if the target is the real aircraft's drag.
7. **Domain/far field:** enlarge to about 50 MAC or add a far-field vortex correction. This is a small effect, about 0.05°.

## Files
- **Plots:** `validation/conv_history.png`, `drift.png`, `mesh_sensitivity.png`, `yplus.png`, `cp_stations.png`, `span_loading.png`, `cp_body_tails.png`.
- **Section notes:** `validation/sec_mesh.md`, `validation/sec_drag.md`.
- **Data:** `validation/data/components.json`, `analysis_*.json`, `mesh_gci.json`, `drift.json`, `drag_buildup.json`, `reference_values.json`, `theory.json`, `split_*.json` and `.trn`, face-data `*.npz`.
- **Scripts:** `extract.py`, `split.py`, `components.py`, `facedata.py`, `analyse.py`, `theory.py`, `solve2.py`, `mesh_sens.py`, `dragbuild.py`, `plot_convergence.py`.
