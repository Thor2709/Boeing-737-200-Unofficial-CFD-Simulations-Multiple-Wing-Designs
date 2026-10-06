# CFD starting-angle estimates

The estimates use the final C5 history alpha, **3.624031° provisional** because `result.json` is absent. The local C5 wing slope is a linear fit to five nearby search angles, using the median of the last five `CL_wing` samples at each angle: **0.055557 CL/deg**. The measured non-wing contribution is `CL - CL_wing = 0.073909`.

At the four basis stations, lift weights are `w = c sqrt(1 - eta^2)`, normalized to sum to one. The required wing lift is target total CL minus the C5 non-wing contribution. The correction is

`alpha0(Ci) = alpha_CFD(C5) + d_geom + d_aero`

`d_geom = sum(w (twist_C5 - twist_i))`

`d_aero = sum(w (alpha0L_i - alpha0L_C5)) + CL_wing_needed (1/a_i - 1/a_C5)`.

Cruise section polars are evaluated in the sweep-normal plane using `Lambda = 25°` at quarter chord: `M_n = M cos(Lambda)`, `alpha_n = alpha / cos(Lambda)`, `cl_n = cl / cos²(Lambda)`, and `(t/c)_n = (t/c) / cos(Lambda)`. The local streamwise section lift demand is converted to `cl_n` before polar CLmax and Cp lookup. A compressible cruise batch uses foil files named `<FOIL>_n25.dat`, with unchanged x-coordinates and y-coordinates scaled by `1/cos(25°)`. Its polar zero-lift angle and slope are normal-plane values and are mapped back as `alpha_0L = alpha_0L,n cos(Lambda)` and `a = a_n cos(Lambda)` before the 3D planform factor. Take-off (Mach 0.2180) remains streamwise with no sweep transform.

No compressible polar directory was supplied for this provisional result. The Mach 0 fallback uses the streamwise foil polars: their fitted angle and slope are converted algebraically to normal-plane values and mapped back, while the section CL samples are scaled by `1/cos²(Lambda)`. The fallback slope uses Prandtl-Glauert at `M_n`, and Cpmin uses Karman-Tsien at `M_n`. The common 3D mapping factor is `a_C5_CFD / weighted_a_2D_C5 = 0.387854`. Spanwise Cp margin is `Cp_n - Cp* (M_n)`; a negative margin marks a section supercritical. No shock correction is applied. The reported band is the four-station correction spread expanded by the weighted linear-extrapolation span where a polar does not reach zero lift; shock uncertainty remains unquantified.

| Case | Current-rule alpha0 (deg) | Corrected alpha0 (deg) | Band (deg) |
|---|---:|---:|---:|
| C1 | 4.402 | 4.261 | 3.672 to 5.212 |
| C2 | 1.695 | 1.652 | 1.058 to 2.603 |
| C3 take-off | 7.453 | 4.623 | -6.676 to 16.880 |
| C4 | 2.829 | 2.888 | 2.517 to 3.450 |

Cruise changes are within 0.14° of the current rule. C3 has one supercritical section and its root's needed section CL (1.958) exceeds the available AOPT polar CLmax (1.839). The AOPT_F zero-lift fit extrapolates about 8.7–9.3° beyond its sampled alpha range, which widens the C3 band. Extend that polar to more negative alpha before relying on the take-off seed.
