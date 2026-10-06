# C5 B737-200 model vs the real aircraft: what is wrong and what it costs (2026-10-05)

Scope: research only. No geometry was changed in this comparison.

Inputs:
- C5 CPU field: `cfd_v2/runs/C5/iH0/final.*.h5`, analysed offline (scratchpad `cmp_fields.py`, `geo_audit.py`, `geo_figs.py`).
- B737-200_forAnsys5.vsp3, `pipeline/aero/build_foils.py`, and the UIUC B737A–D coordinates.
- Literature (sources at the end).

Figures: `research/figs_geo_disc/span_loading_twist.png`, `section_waviness.png`.

## 0. The gap

C5 at M 0.74, CL 0.4895 (untrimmed, Sref 91.04 m²):

| Source | CD [counts] | L/D |
|---|---|---|
| **C5 CFD (CPU, prod mesh)** | **479** | 10.2 |
| xlsx target (cd0 324 + cdi 108) | 432 | 11.33 |
| Real 737-200: Krull regression of Boeing polar data | ≈ 318 + small wave drag ≈ 320–335 | ≈ 15 |

Derivation of the real-aircraft value:
- Formula: CD = 199.09 + 404.58·CL² / (1 − 0.01185·(M/0.3 − 1)^7.074), in counts.
- At M 0.74, CL 0.4895: 199 + 118 = 317.
- The B737-200 wave-drag term at Mcrit is tiny (0.04 counts).
- The figure axes run CD 210–330 over M 0.45–0.85 for CL 0.2–0.4.
- d = 404.58 implies e ≈ 0.89 with AR 8.83, so the reference is the 91.04 m² trapezoid.
- CL 0.49 lies slightly above the fitted CL range (0.2–0.4).

**Conclusions:**
- Our CFD is about 150 counts (≈ 45 %) above the real aircraft.
- The xlsx target itself sits about 100 counts above the real aircraft. The xlsx is the project's authority (PLAN §0), but it should not be read as "the real 737".

## 1. Ranked discrepancies

Drag numbers are estimates.
- **Excess** is CFD minus a realistic value for the real aircraft.
- Component values come from C5 surface integration in counts, on full-aircraft Sref.

### 1. Wing aerodynamic design at M 0.74 (largest; est. +80 to +100 counts)

**Evidence.**
- Wing CDp is 218 counts. A real wing at this CL should carry about 97 counts of induced drag (e 0.89), plus roughly 20 counts of profile pressure drag and a few counts of wave drag.
- The upper-surface shocks reach isentropic Mach 1.4–1.6, which is shock-induced-separation strength. The real cruise wing is designed for about 1.2.
- 7.2 m² (9.7 %) of the wing has reverse flow.
- The CL-α slope at the operating point has dropped to about 0.05/deg, which means the wing sits on its lift break.
- Korn Mdd along the span is 0.72–0.745, so we cruise at or above drag divergence.

**Causes.**

a. **No washout.**
- The model's local incidence is +0.7° inboard and +0.4° outboard (chord line measured on the mesh, ±0.2°).
- The raw UIUC sections carry embedded incidence: +1.01° (A, root), +0.50° (B), 0 (C), 0 (D). That is about 1° of washout.
- `build_foils.py normalise()` derotated them, and the VSP twist is 'basis' (TWIST_BASIS 1.0), so that washout was lost.
- A rigid model also misses the cruise aeroelastic wash-out of a real swept wing. Its size for the 737 is not published; order 1–2° for swept transport wings.

b. **Outboard overload.**
- c·cl against an elliptic loading with the same lift:
  - η 0.3–0.5: 0.89–0.90;
  - η 0.72: 1.09;
  - η 0.84: 1.12;
  - η 0.96: 1.5.
- Section cl peaks at 0.58–0.62 at η 0.72–0.80, against 0.43–0.46 inboard.
- The outboard sections are the thinnest (10–11 %) yet carry the highest cl, so they get the strongest shocks and the separation.
- Causes (a) and (b) together act like ~1–2° of missing twist.

c. **Sections are low-fidelity reconstructions.**
- The real wing uses BAC 449/450/451 (root/inboard) and BAC 442 (tip). These are proprietary.
- The UIUC "Boeing 737 root/midspan/outboard" files have only 23 points per surface, and two files carry the same "midspan" label.
- The thickness distribution is non-monotonic: A 15.4 % (root), B 12.6 %, C 10.0 %, D 10.8 % (tip).
- Curvature: A, B and D are smooth at this resolution; C has a local bump at x/c 0.45–0.6.
- The initial calculation ("13–17 curvature sign changes") was a parsing artefact. The sections are coarse, not strongly wavy.
- The station mapping (root, 4.72, 9.45 and 14.17 m) is our assumption; UIUC gives no stations.

d. **Missing yehudi (inboard trailing-edge extension).**

| | Real | Our model |
|---|---|---|
| Root chord | 7.32 m | 4.79 m |
| Gross area | 102 m² | 91 m² |
| MAC | 3.80 m | 3.47 m |

- On the real wing the yehudi lowers the root t/c at the same depth and unloads the inboard sections in cl terms.
- Ours runs a 15.4 % root at cl 0.43–0.53, past its Korn Mdd. This causes the inboard shock and contributes to the junction separation.

### 2. Nacelle / inlet (est. +40 to +50 counts)

- Inlet duct diameter is about 0.91 m, against the JT8D fan at 1.37 m (max nacelle width 1.50 m). With highlight r ≈ 0.47–0.50 m and max r ≈ 0.72 m, the lip is very thick and blunt.
- Engine cowl total is CDp 65.5 + CDv 5.2 counts:
  - The front 0.5 m alone carries 55 counts.
  - Cp reaches +1.46, above the stagnation value of 1.147. This is a numerical/geometry anomaly at the lip.
  - Supersonic overspeed: Cp −0.84 against Cp* −0.63.
  - 0.86 m² of reverse flow on the cowl.
- Duct Mach is about 0.65 and the mass-flow ratio about 0.9, so the drag is not spillage-dominated; it comes from the lip shape.
- Bookkeeping: a real drag polar excludes the internal duct drag (thrust/drag bookkeeping). Our flow-through nacelle includes it.
- Pylon adds 5.5 counts. The spanwise loading dips sharply at η ≈ 0.32 (pylon station), from pylon/nacelle-wing interference.

### 3. Fuselage and wing-body junction (est. +30 to +60 counts, least certain)

- Fuselage CDp is 86.5 counts. A real fuselage has about 15–25 counts of form drag, plus its share of the induced drag.
- Breakdown by x-bin:
  - nose 0–6 m: net +31 (113 forward, 82 suction);
  - wing junction 9–15 m: +24;
  - 18–24 m: +17;
  - aft 24–30 m: +15;
  - 1.2 m² of reverse flow at the tail-cone tip.
- There is no wing-body belly fairing, only a 2.41 m² fillet ("fairing" zone, 6.6 counts). The real 737-200 has a fairing over the wing box and gear bays that controls the junction pressure gradients.
- Overall dimensions match the real aircraft: length 29.54, width 3.76, height 4.01 m. The nose and tail-cone shape are not verified against drawings.

### 4. Tail planforms and trim (drag est. +5 to +15 counts; dominant for CM)

| | Real | Our model |
|---|---|---|
| Htail span | 10.97 m | 8.73 m |
| Htail sweep | 30° | 25.4° |
| Htail AR | 4.15 | ≈ 2.6 |
| Htail dihedral | 7° | — |
| Htail section | — | NACA 0012 at 0° |
| Fin height above fuselage | ≈ 5.9 m | ≈ 4.6 m |
| Fin sweep | 35° | 28.7° |
| Fin AR | 1.64 | ≈ 1.05 |
| Fin area | 20.8 m² | 20.8 m² (matched) |

- Low-AR tails have a lower lift-curve slope, which changes stability and CM.
- The real stabiliser is trimmable; ours is fixed at 0°, and C5 is untrimmed.
- Component drag: htail CDp −8 + CDv 10.4; fin CDp 14.4 + CDv 8.6.

### 5. CFD-side offsets (not geometry; they mask items 1–4)

**Friction under-predicted** (est. −50 to −60 counts):
- total CDv is 91 counts;
- flat-plate estimates: wing ≈ 43 (CFD 28), fuselage ≈ 62 (CFD 35);
- y+ median ≈ 36–41 with wall functions on a 1M-cell mesh.

**Numerics:**
- CPU vs GPU gives about 10 % CL at the same α (T2 note). This is the transonic lift-break sensitivity.
- g400 vs prod: CL +1.6 %, CD +9.7 %.
- The real-wing fixes in items 1–3 would weaken the shocks and should shrink both sensitivities.

**Balance:**
- 479 ≈ 330 (real) + 90 (wing) + 45 (nacelle) + 45 (fuselage) + 10 (tails) − 55 (friction) + bookkeeping/noise. All terms are estimates.
- The same balance explains why CFD is "only" 11 % above the xlsx target: the pressure-drag excess is partly cancelled by under-predicted friction, and the target itself is high.

## 2. What would fix the most, in order (proposals; no edits made)

1. **Restore washout:**
   - Keep the source incidence of the UIUC sections, i.e. stop derotating in `normalise()` or put it into VSP twist.
   - Optionally add 1–2° of aeroelastic cruise washout.
   - Expected to unload the outboard wing and weaken the outboard shocks.
2. **Add the yehudi:** inboard trailing-edge extension to a 7.32 m root chord, with section t/c re-scaled. Keep Sref 91.04 for the coefficients.
3. **Real inlet:** fan-sized highlight (~1.25–1.37 m) and a thin JT8D-style lip, with internal duct drag excluded in the bookkeeping.
4. **Belly fairing** over the wing box.
5. **Tails to real planform:** htail span 10.97 m, AR 4.15, sweep 30°, dihedral 7°; fin AR 1.64, sweep 35°. Then run the trimmed case.
6. **Higher-fidelity sections:** Resample to ≥ 100 points per surface and check monotonic t/c.
7. **CFD side:** y+ ≈ 1 near-wall resolution on the wing (cell budget permitting), or report friction with a flat-plate correction flag.

## Sources

- [b737.org.uk detailed tech data](http://www.b737.org.uk/techspecsdetailed.htm): wing, tail, fuselage, yehudi root chord.
- [Jenkinson/Elsevier aircraft data, table 2](https://booksite.elsevier.com/9780340741528/appendices/data-a/table-2/table.htm): reference area, MAC, t/c, tail data.
- [Krull, Generic drag polars, HAW Hamburg master thesis](https://www.fzt.haw-hamburg.de/pers/Scholz/arbeiten/TextKrullMaster.pdf): B737-200 polar regression (Table 7.1, Figure 7.5).
- [Lednicer, Incomplete Guide to Airfoil Usage](https://m-selig.ae.illinois.edu/ads/aircraft.html): 737-100/-200 BAC 449/450/451 root, BAC 442 tip.
- [UIUC airfoil coordinates database](https://m-selig.ae.illinois.edu/ads/coord_database.html): B737A–D.
- [aircraftinvestigation.info 737-200](https://aircraftinvestigation.info/airplanes/737-200.html): CD0 0.0259, L/D max 14.1 (low credibility).
