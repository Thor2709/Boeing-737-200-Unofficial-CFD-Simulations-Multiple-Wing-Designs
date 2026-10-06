"""P3 driver: build the fluid-domain .dsco for one case and one horizontal-tail incidence.

  Python312\python.exe build_domain.py C1 [--ih 0]

1. (CPython 3.13 / OpenVSP) vsp_case.py  -> geom_v2/<case>_iH<ih>/{case.vsp3, <case>_wing.stp/.igs, vsp_info.json}
                                            geom_v2/common/{htail_iH<ih>,vtail,nacelle,pylon}.stp  (shared parts, exported once)
2. (CPython 3.12 / Discovery 2026 R1 via PyAnsys Geometry) dom_lib.build_case -> cfd_v2/geom/<case>_domain.dsco
   (case_iH<ih>_domain.dsco when ih != 0) + geom_v2/<case>_iH<ih>/domain_report.json
Only ONE Ansys product may run: close any Discovery session before starting.
"""
import os, sys, argparse, subprocess
from pathlib import Path
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument('case', choices=['C1', 'C2', 'C3', 'C4', 'C5']); ap.add_argument('--ih', type=float, default=0.0)
a = ap.parse_args()
try:
    from b737wing.config import REPO_ROOT
    ROOT = str(REPO_ROOT)
except ImportError:
    ROOT = str(Path(__file__).resolve().parents[4])
PYTHON = os.environ.get('PYTHON') or sys.executable
tag = ('%g' % a.ih).replace('-', 'm').replace('.', 'p')
if not os.path.exists(os.path.join(ROOT, 'geom_v2', f'{a.case}_iH{tag}', 'vsp_info.json')):
    subprocess.check_call([PYTHON, os.path.join(HERE, 'vsp_case.py'), a.case, '--ih', str(a.ih)])
os.environ.setdefault("AWP_ROOT261", r"C:\Program Files\ANSYS Inc\ANSYS Student\v261")
import ansys.geometry.core as ag
from ansys.geometry.core import launch_modeler_with_discovery
from ansys.geometry.core.connection.backend import ApiVersions
m = launch_modeler_with_discovery(version=261, api_version=ApiVersions.V_261, hidden=False, timeout=900)
try:
    ns = {'m': m, 'ag': ag, 'HERE': HERE, '__file__': os.path.join(HERE, 'dom_lib.py')}
    exec(open(os.path.join(HERE, 'dom_lib.py'), encoding='utf-8').read(), ns)
    ns['build_case'](a.case, a.ih, log=lambda s: print(s, flush=True))
finally:
    m.close()
