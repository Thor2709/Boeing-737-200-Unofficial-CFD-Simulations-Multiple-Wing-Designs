"""Persistent Discovery session for P3 development (python 3.12 + ansys-geometry-core).
Executes <work>/q/cmd_*.py (namespace: m, ag) and writes cmd_*.out. Start in the background; stop by creating q/STOP."""
import os, sys, glob, time, traceback, io, contextlib
os.environ.setdefault("AWP_ROOT261", r"C:\Program Files\ANSYS Inc\ANSYS Student\v261")
import ansys.geometry.core as ag
from ansys.geometry.core import launch_modeler_with_discovery
from ansys.geometry.core.connection.backend import ApiVersions
Q = os.path.join(sys.argv[1], 'q')
os.makedirs(Q, exist_ok=True)
if os.path.exists(os.path.join(Q, 'STOP')):
    raise RuntimeError('remove the stale Discovery STOP marker after checking the prior session')
if glob.glob(os.path.join(Q, 'cmd_*.py')):
    raise RuntimeError('pending Discovery commands exist; inspect them before starting a new session')
m = launch_modeler_with_discovery(version=261, api_version=ApiVersions.V_261, hidden=False, timeout=900)
GEOM_DIR = os.path.dirname(os.path.abspath(__file__))
ns = {'m': m, 'ag': ag, 'HERE': GEOM_DIR, '__file__': os.path.join(GEOM_DIR, 'dom_lib.py')}
open(os.path.join(Q, 'READY'), 'w').write('ready')
try:
    while not os.path.exists(os.path.join(Q, 'STOP')):
        for f in sorted(glob.glob(os.path.join(Q, 'cmd_*.py'))):
            out = f[:-3] + '.out'
            code = open(f, encoding='utf-8').read(); os.rename(f, f + '.done')
            class Tee:
                def __init__(s): s.f = open(out, 'w', encoding='utf-8', buffering=1)
                def write(s, t): s.f.write(t); s.f.flush()
                def flush(s): s.f.flush()
            buf = Tee()
            try:
                with contextlib.redirect_stdout(buf): exec(compile(code, f, 'exec'), ns)
            except BaseException:
                buf.write(traceback.format_exc())
                buf.write('\n<<FAILED>>')
            else:
                buf.write('\n<<DONE>>')
            buf.f.close()
        time.sleep(1)
finally:
    try: m.close()
    except Exception: pass
