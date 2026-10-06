"""Live API smoke test of FluentBackend on a small mesh (few iterations, every call path).

python -m b737wing.cfd.solve.smoke_live <mesh> <out_dir> [case]
"""
import json
import sys
import time
import traceback
from pathlib import Path

from b737wing.cfd.solve.conditions import condition_for_case
from b737wing.cfd.solve.fluent_io import FluentBackend, launch_fluent

mesh, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
case = sys.argv[3] if len(sys.argv) > 3 else "C5"
out.mkdir(parents=True, exist_ok=True)
steps = []
backend = FluentBackend(launch_fluent(4, str(out)))
try:
    for name, fn in [
        ("configure", lambda: backend.configure(str(mesh), condition_for_case(case))),
        ("set_alpha", lambda: backend.set_alpha(2.5)),
        ("courant", lambda: backend.set_courant(2.0)),
        ("iterate", lambda: backend.iterate(10)),
        ("sample", lambda: backend.sample()),
        ("export_walls", lambda: backend.export_walls(str(out / "walls.csv"))),
        ("save_final", lambda: backend.save_final(str(out))),
        ("export_ensight", lambda: backend.export_ensight(str(out / "ensight"))),
    ]:
        t = time.time()
        try:
            r = fn()
            steps.append({"step": name, "ok": True, "s": round(time.time() - t, 1),
                          "result": r if isinstance(r, (dict, list, float, int, str)) else None})
        except Exception:
            steps.append({"step": name, "ok": False, "error": traceback.format_exc()[-1500:]})
            break
finally:
    try:
        backend.close()
    except Exception:
        pass
(out / "smoke.json").write_text(json.dumps(steps, indent=1, default=str), encoding="utf-8")
print(json.dumps(steps, indent=1, default=str)[-4000:])
