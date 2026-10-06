# Numerical & CFD Methods

This document describes the numerical methods and analysis pipeline for the Boeing 737-200 wing aerodynamic evaluation.

## 1. Geometry Chain: OpenVSP → STEP → Ansys Discovery
- **CAD & Aerodynamic Modeling:** Aircraft geometries are parameterized and generated in OpenVSP (`.vsp3` models) including the fuselage, main wing, nacelles, pylons, vertical fin, and horizontal stabilizer.
- **Export & Fluid Domain:** Models are exported from OpenVSP as STEP solid geometry. In Ansys Discovery, a computational half-model fluid domain is created with bounding dimensions sized to 31–47 MAC (Mean Aerodynamic Chord) to preserve undisturbed pressure far-field boundary conditions without artificial blockage.

## 2. Aerofoil Repanelling & Blunt Trailing Edge
- **Source Coordinate Processing:** Raw aerofoil coordinates (e.g. Selig/Lednicer formats from UIUC database) are normalized and filtered for monotonicity in [`src/b737wing/aero/foils.py`](../src/b737wing/aero/foils.py).
- **Curvature-Weighted Repanelling:** [`repanel()`](../src/b737wing/aero/foils.py) constructs arc-length cubic splines on upper and lower surfaces, clustering nodes at high-curvature regions (leading edge) and resolving suction peaks using curvature-based density distributions.
- **Blunt Trailing Edge:** [`blunt_te()`](../src/b737wing/aero/foils.py) introduces a finite trailing-edge thickness gap $\Delta y \propto x^2$ in the chord frame, preventing prism collapse and negative-volume cell skewness during volume meshing.
- **Sweep Transformation:** For swept-wing analysis, section coordinates, Mach numbers ($M_n = M \cos\Lambda$), and thickness-to-chord ratios are mapped to the sweep-normal plane (`write_sweep_normal_foils()` in [`src/b737wing/flow5/xmlgen.py`](../src/b737wing/flow5/xmlgen.py)).

## 3. flow5 Two-Pass Workflow
- **Pass 1 (2D Viscous Polars):** Batch isolated aerofoil analyses are executed across Reynolds numbers ($2\times 10^6$ to $5\times 10^7$) and angles of attack using XFoil integration ([`src/b737wing/flow5/xmlgen.py`](../src/b737wing/flow5/xmlgen.py)).
- **Pass 2 (3D Panel Method):** 3D wing-alone configurations are solved using 3D panel methods (e.g., `TRIUNIFORM` / `QUADS` formulation, flat panel wake progression) with section viscous drag and separation limits interpolated from Pass 1 polar databases ([`src/b737wing/flow5/runner.py`](../src/b737wing/flow5/runner.py), [`src/b737wing/flow5/post.py`](../src/b737wing/flow5/post.py)).

## 4. Meshing & The 1M-Cell Limit
- **Poly-Hexcore Topology:** Generated using Ansys Fluent Meshing via [`src/b737wing/cfd/mesh/mesh_case.py`](../src/b737wing/cfd/mesh/mesh_case.py). Hexcore elements fill the bulk domain, transitioning through polyhedral buffer layers to prism boundary layers on wall zones.
- **Prism Layers & Wall Sizing:** 12 prism layers with an initial height of $4\times 10^{-4}\text{ m}$ (first cell $y^+ \sim 20$–$40$) and growth rate 1.2 ensure robust near-wall blending with $k$-$\omega$ SST turbulence modeling.
- **Body of Influence (BOI):** Refinement boxes (`write_boxes()`) enclose the aircraft wing-body envelope, wake region, and pylon/nacelle junction channel.
- **License Constraint:** Strict cell budgeting restricts production half-model meshes to < 1,048,576 cells (`CAP = 1_048_576` in [`mesh_case.py`](../src/b737wing/cfd/mesh/mesh_case.py)), requiring local sizing scaling to maintain mesh quality (min orthogonal quality $> 0.02$).

## 5. Steady Solver & Target $C_L$ Secant Search
- **CPU Density-Based Implicit Formulation:** Transonic cruise ($M = 0.74$) solved using density-based Roe flux-difference splitting and double precision ([`src/b737wing/cfd/solve/run_case.py`](../src/b737wing/cfd/solve/run_case.py)).
- **Courant Ramp:** CFL is ramped gradually from 2.0 (first 100 iterations) to 5.0 (300 iterations) up to 10.0–20.0 to stabilize shock formation and prevent early divergence.
- **Secant Search Algorithm:** Target lift coefficient ($C_L \approx 0.4895$) is achieved via an automated $\alpha$ secant iteration ([`src/b737wing/cfd/solve/search.py`](../src/b737wing/cfd/solve/search.py)). Initial $\alpha_0$ seeds are derived from flow5 predictions, followed by steps guarded within $0.02 \le dC_L/d\alpha \le 0.20 \text{ deg}^{-1}$ and capped at $1.0^\circ$ increments.
- **Exponential Settle Criterion:** Because low-speed aft-body and tail-root separations exhibit long relaxation timescales ($\tau \sim 500$–$600$ iterations), convergence is determined by a 3-point exponential extrapolation window (`_fit_settle_window()` in [`search.py`](../src/b737wing/cfd/solve/search.py)), requiring extrapolated remaining change $|y - y_\infty| < 5\times 10^{-4}$ for $C_L$, $0.5$ count for $C_D$, and $10^{-3}$ for $C_m$.

## 6. Grid Convergence Index (GCI) Method
- **Roache Verification Framework:** Evaluates spatial discretization uncertainty across systematically refined grid levels (coarse, medium, fine) at constant flight conditions.
- **Effective Order & Error Bands:** Computes apparent order of convergence $p = \ln\left((f_3 - f_2)/(f_2 - f_1)\right) / \ln(r)$, Richardson extrapolation to zero grid spacing, and GCI uncertainty bands with safety factor $F_s = 1.25$ (three grids) or $3.0$ (two grids), quantifying discretization error on $C_L$, $C_D$, and $C_m$.

## 7. Transient SBES on GPU
- **In-Session Steady GPU Warm-Start:** Native pressure-based coupled GPU solver runs 100 steady iterations within the active session to initialize flow fields and prevent energy equation divergence.
- **Transient Formulation:** Switches to pressure-based SIMPLEC with under-relaxation factors ($p=0.3$, momentum $=0.5$, $\rho=0.5$, $T=0.7$, $k$-$\omega=0.6$) and 1st-order/bounded unsteady formulation.
- **Stress-Blended Eddy Simulation (SBES):** Viscous model activates SBES hybrid RANS-LES (`setup.models.viscous.hybrid_rans_les = "stress-blended-eddy-simulation"`), resolving unsteady separated shear layers behind transonic shocks.
- **Time Stepping:** Time step size $\Delta t = 1.0\times 10^{-4}\text{ s}$ (matching shock-cell acoustic CFL constraints) with 15 inner iterations per time step.

## 8. Post-Processing & Aero Analytics
- **Surface Force Breakdown:** [`src/b737wing/post/force_breakdown.py`](../src/b737wing/post/force_breakdown.py) integrates wall pressure and shear stress using exact polygon Newell area vectors, decomposing lift and drag into component zones (wing, fuselage, htail, vtail, nacelle lip, cowl, pylon).
- **$C_p$ Section Distributions:** Chordwise surface pressure distributions at span stations ($\eta = 0.15, 0.30, 0.50, 0.70, 0.90$) extract shock locations, suction peaks, and trailing-edge pressure recovery.
- **Breguet Range Evaluation:** [`src/b737wing/post/breguet.py`](../src/b737wing/post/breguet.py) computes cruise mission range in nautical miles from operating $C_L$, $C_D$, and aircraft weight schedules ($W_0 = 115,500\text{ lbf}$, $W_1 = 83,474\text{ lbf}$, $\text{TSFC} = 1.662\times 10^{-4}\text{ s}^{-1}$).
- **EnSight & ParaView Visualization:** Transient volumetric fields exported to EnSight Gold format are rendered with batch ParaView scripts (`pvbatch`) for Q-criterion iso-surfaces, Mach slice contours, and surface streamlines.
