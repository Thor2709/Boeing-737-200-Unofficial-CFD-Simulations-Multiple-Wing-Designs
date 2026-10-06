"""CPU solver core-scaling benchmark (owner request 2026-10-04: 4-8-12-16-20 cores).

python solver_scaling.py <mesh.msh.h5> <out_dir> [cores ...]
Per core count: launch Fluent solver (double), density-based implicit, SST, ideal gas,
Sutherland, M0.74 far field at alpha 2 deg, hybrid init, then iterations in chunks of 5 for
at least RUN_S = 420 s wall (well beyond the Intel PL1 tau turbo window, typically 28-128 s,
and long enough for thermal steady state). Every chunk is logged to <out>/np<n>/chunks.csv.
Reported: s/iter in the first 60 s (turbo) and over the last 50 % of the run (sustained,
used for the decision). 120 s cool-down between core counts so each starts from the same
state. Core placement is done by b737wing/tools/pin_pcores.py (P-cores first).
Writes <out_dir>/scaling.csv and prints one line per run.
"""
import csv, math, os, sys, time
from b737wing.config import AWP_ROOT261
os.environ["AWP_ROOT261"] = str(AWP_ROOT261)
import ansys.fluent.core as pyfluent

mesh, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
cores = [int(c) for c in sys.argv[3:]] or [4, 8, 12, 16, 20]
os.makedirs(out, exist_ok=True)
CHUNK, RUN_S, COOL_S = 5, 420.0, 120
rows = []
for n in cores:
    run = os.path.join(out, f"np{n}"); os.makedirs(run, exist_ok=True)
    t0 = time.time(); row = {"cores": n}
    s = None
    for attempt in range(3):  # gRPC 'connection reset' on launch is transient (seen in P4)
        try:
            s = pyfluent.launch_fluent(mode="solver", precision="double", processor_count=n,
                                       product_version="26.1.0", cleanup_on_exit=True,
                                       start_watchdog=False, cwd=run)
            break
        except Exception as e:
            row["error"] = f"launch: {e!r}"[:300]; time.sleep(30)
    if s is None:
        rows.append(row); print(row, flush=True); continue
    row.pop("error", None)
    try:
        st = s.settings; S = st.setup
        t1 = time.time(); st.file.read_mesh(file_name=mesh); row["read_s"] = round(time.time() - t1, 1)
        S.general.solver.type = "density-based-implicit"
        S.models.energy.enabled = True
        S.models.viscous.model = "k-omega"; S.models.viscous.k_omega_model = "sst"
        air = S.materials.fluid["air"]; air.density.option = "ideal-gas"; air.viscosity.option = "sutherland"
        S.general.operating_conditions.operating_pressure = 0
        pff = S.boundary_conditions.pressure_far_field["farfield"]
        pff.momentum.gauge_pressure = 30090; pff.momentum.mach_number = 0.74; pff.thermal.temperature = 228.7
        a = math.radians(2.0)
        pff.momentum.flow_direction = [{"option": "value", "value": math.cos(a)}, {"option": "value", "value": 0},
                                       {"option": "value", "value": math.sin(a)}]
        t1 = time.time(); st.solution.initialization.hybrid_initialize(); row["init_s"] = round(time.time() - t1, 1)
        chunks = []; t1 = time.time(); it = 0
        with open(os.path.join(run, "chunks.csv"), "w", newline="") as cf:
            cw = csv.writer(cf); cw.writerow(["t_s", "iters", "s_per_iter"])
            while time.time() - t1 < RUN_S:
                c0 = time.time(); s.solution.run_calculation.iterate(iter_count=CHUNK); dt = time.time() - c0
                it += CHUNK; chunks.append((c0 - t1, dt / CHUNK)); cw.writerow([round(c0 - t1, 1), it, round(dt / CHUNK, 4)]); cf.flush()
        first = [v for t, v in chunks if t < 60] or [chunks[0][1]]
        late = [v for t, v in chunks if t >= RUN_S / 2]
        row["s_per_iter_turbo"] = round(sum(first) / len(first), 3)
        row["s_per_iter"] = round(sum(late) / len(late), 3)
        row["iters"] = it
    except Exception as e:
        row["error"] = repr(e)[:300]
    finally:
        try: s.exit()
        except Exception: pass
    row["total_s"] = round(time.time() - t0, 1)
    rows.append(row); print(row, flush=True)
    time.sleep(COOL_S)  # licence/MPI clear-up and the same thermal start state
fields = ["cores", "read_s", "init_s", "s_per_iter_turbo", "s_per_iter", "iters", "total_s", "error"]
with open(os.path.join(out, "scaling.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
base = next((r["s_per_iter"] for r in rows if r.get("cores") == 4 and "s_per_iter" in r), None)
for r in rows:
    if base and "s_per_iter" in r:
        print(f"{r['cores']:>2} cores: {r['s_per_iter']:.3f} s/iter sustained (turbo {r['s_per_iter_turbo']:.3f}), speed-up {base / r['s_per_iter']:.2f}x vs 4")
