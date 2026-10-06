"""P3 step 1: build the case-specific VSP model and export per-component STEP solids.

Run with CPython 3.13 (OpenVSP 3.53.1):
  Python313\\python.exe vsp_case.py <C1..C5> [--ih DEG] [--out DIR]

Source model B737-200_forAnsys5.vsp3 is READ-ONLY; the modified copy is written to <out>/case.vsp3.
Only the main wing changes between cases (sections / aerofoils / twist). Everything else (tails, nacelle, pylon)
is exported once into geom_v2/common and shared by all cases; the horizontal tail depends on iH only
(geom_v2/common/htail_iH<ih>.stp).

iH convention: positive = leading edge up (nose-up incidence), rotation about the y axis through the
25 % point of the HT root chord (the point stays fixed, the HT pivots about it).
"""
import sys, os, json, math, argparse, shutil
from pathlib import Path
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from vsp_init import *   # noqa  (vsp)

try:
    from b737wing.config import REPO_ROOT
    ROOT = str(REPO_ROOT)
except ImportError:
    ROOT = str(Path(__file__).resolve().parents[4])
SRC = os.path.join(ROOT, 'B737-200_forAnsys5.vsp3')
FOILS = os.path.join(ROOT, 'aerofoils_v2')

# ---- planform constants (flow5_v2/basis/basis_sections.md; identical to pipeline/flow5/cases.py) ----
DIH = 6.0
C_ROOT, C_TIP, Y_TIP = 4.79384, 1.62991, 14.09556          # projected semispan
SECT_Y = [0.0, 4.69852, 9.39704, 14.09556]
SECT_C = [4.79384, 3.73919, 2.68455, 1.62991]
# Flap inboard edge: the brief gives y = 1.88 m (fuselage side). A step exactly at the hull (half width 1.88 m) made the
# Discovery union with the fuselage fail (coincident step faces), so the 0.02 m AOPT->AOPT_F transition is placed INSIDE the
# fuselage at y = 1.40 m; the exposed flap still starts at the fuselage side. Outer edge 0.74*semispan as in flow5.
FLAP_INNER_Y, FLAP_OUTER_Y, FLAP_GAP = 1.40, 0.74 * Y_TIP, 0.02
TWIST_BASIS = 1.0
TW_LIN_ROOT, TW_LIN_TIP = -0.459177, 0.459175

CASES = {
    'C1': dict(foils=['A0012'] * 4, twist='basis'),
    'C2': dict(foils=['AOPT'] * 4, twist='basis'),
    'C3': dict(foils=['AOPT'] * 4, twist='basis', flap=True),
    'C4': dict(foils=['AOPT'] * 4, twist='linear'),
    'C5': dict(foils=['B737A', 'B737B', 'B737C', 'B737D'], twist='basis'),
}


def chord_at(y):
    return C_ROOT + (C_TIP - C_ROOT) * y / Y_TIP


def twist_at(case, y):
    if case['twist'] == 'linear':
        return TW_LIN_ROOT + (TW_LIN_TIP - TW_LIN_ROOT) * y / Y_TIP
    return TWIST_BASIS


def wing_sections(name):
    """[(y_proj, chord, twist_deg, foil)] for the case; flap edges as in flow5 xmlgen (sharp 0.02 m transitions)."""
    case = CASES[name]
    secs = [(y, c, twist_at(case, y), f) for y, c, f in zip(SECT_Y, SECT_C, case['foils'])]
    if case.get('flap'):
        keep = [s for s in secs if not (FLAP_INNER_Y < s[0] < FLAP_OUTER_Y)]
        sp = [(FLAP_INNER_Y, 'AOPT'), (FLAP_INNER_Y + FLAP_GAP, 'AOPT_F'),
              (FLAP_OUTER_Y - FLAP_GAP, 'AOPT_F'), (FLAP_OUTER_Y, 'AOPT')]
        keep += [(y, chord_at(y), twist_at(case, y), f) for y, f in sp]
        secs = sorted(keep, key=lambda s: s[0])
    return secs


def planform_metrics(secs, x0):
    """One-side planform (projected y) area, span, MAC and the 25 % MAC x position (global x)."""
    tan = math.tan(math.radians(25.0))
    xle = lambda y: 0.25 * C_ROOT + y * tan - 0.25 * chord_at(y)     # LE x offset: c/4 locus swept 25 deg
    c0, c1 = C_ROOT, C_TIP
    area = 0.5 * (c0 + c1) * Y_TIP                                   # chord is linear over the whole wing
    mac = (c0 * c0 + c0 * c1 + c1 * c1) / 3.0 / (0.5 * (c0 + c1))   # int c^2 dy / int c dy
    ymac = (C_ROOT - mac) / ((C_ROOT - C_TIP) / Y_TIP)               # station where chord(y) = MAC
    x_le_mac = x0 + xle(ymac)
    return dict(area_half_projected_m2=area, span_proj_full_m=2 * Y_TIP, semispan_proj_m=Y_TIP,
                semispan_surface_m=Y_TIP / math.cos(math.radians(DIH)),
                MAC_m=mac, y_MAC_m=ymac, x_LE_MAC_m=x_le_mac, x_25MAC_m=x_le_mac + 0.25 * mac,
                S_full_projected_m2=2 * area)


def gp(g, n, grp='XForm'):
    return vsp.GetParmVal(g, n, grp)


def find_geoms():
    out = {}
    for g in vsp.FindGeoms():
        n = vsp.GetGeomName(g)
        key = ('fuselage' if 'Fuselage - Front' in n else 'wing' if 'Main Wing' in n else 'htail' if 'Horizontal' in n
               else 'vtail' if 'Vertical' in n else 'nacelle' if 'Nacelle' in n else 'pylon')
        out[key] = g
    return out


def load_foil(xsec, name):
    p = os.path.join(FOILS, f'{name}_bluntTE.dat')
    vsp.ReadFileAirfoil(xsec, p)
    # keep the blunt trailing edge of the .dat: no automatic TE closing
    vsp.SetParmVal(vsp.GetXSecParm(xsec, 'TE_Close_Type'), 0.0)


def set_wing(g, secs):
    xs = vsp.GetXSecSurf(g, 0)
    n = len(secs)
    while vsp.GetNumXSec(xs) < n:
        vsp.InsertXSec(g, 1, vsp.XS_FILE_AIRFOIL)
    while vsp.GetNumXSec(xs) > n:
        vsp.CutXSec(g, 1)
    vsp.Update()
    cd = math.cos(math.radians(DIH))
    for i in range(1, n):
        vsp.SetDriverGroup(g, i, vsp.SPAN_WSECT_DRIVER, vsp.ROOTC_WSECT_DRIVER, vsp.TIPC_WSECT_DRIVER)
    for i in range(n):
        x = vsp.GetXSec(xs, i)
        y, c, tw, foil = secs[i]
        load_foil(x, foil)
        sp = lambda nm, v: vsp.SetParmVal(vsp.GetXSecParm(x, nm), v)
        sp('Twist', tw); sp('Twist_Location', 0.25)
        if i >= 1:
            y0, c0 = secs[i - 1][0], secs[i - 1][1]
            sp('Sweep', 25.0); sp('Sweep_Location', 0.25); sp('Dihedral', DIH)
            sp('Span', (y - y0) / cd); sp('Root_Chord', c0); sp('Tip_Chord', c)
    vsp.Update()


def set_tail_foils(g):
    xs = vsp.GetXSecSurf(g, 0)
    for i in range(vsp.GetNumXSec(xs)):
        load_foil(vsp.GetXSec(xs, i), 'A0012')   # tails are NACA 0012: same 0.25 % c blunt TE as the wing .dat files
    vsp.Update()


def set_ih(g, ih):
    """Rotate the HT about the y axis through its root 25 % chord point; ih > 0 = leading edge up."""
    L = [gp(g, 'X_Rel_Location'), gp(g, 'Y_Rel_Location'), gp(g, 'Z_Rel_Location')]
    xs = vsp.GetXSecSurf(g, 0)
    c = vsp.GetParmVal(vsp.GetXSecParm(vsp.GetXSec(xs, 1), 'Root_Chord'))
    th = math.radians(ih)
    vsp.SetParmVal(g, 'Y_Rel_Rotation', 'XForm', ih)
    vsp.SetParmVal(g, 'X_Rel_Location', 'XForm', L[0] + 0.25 * c * (1 - math.cos(th)))
    vsp.SetParmVal(g, 'Z_Rel_Location', 'XForm', L[2] + 0.25 * c * math.sin(th))
    vsp.Update()
    return c


def pt(g, u, w):
    p = vsp.CompPnt01(g, 0, u, w)
    return [p.x(), p.y(), p.z()]


def export_step(parts, key, out, iges=None):
    errs = vsp.ErrorMgrSingleton.getInstance()
    for gg in parts.values():
        vsp.SetSetFlag(gg, 3, gg == parts[key])
    if os.path.exists(out):
        os.remove(out)
    vsp.SetAnalysisInputDefaults('SurfaceIntersection')
    vsp.SetIntAnalysisInput('SurfaceIntersection', 'SelectedSetIndex', [3])
    vsp.SetIntAnalysisInput('SurfaceIntersection', 'CADLenUnit', [vsp.LEN_M])
    vsp.SetIntAnalysisInput('SurfaceIntersection', 'STEPFileFlag', [1])
    vsp.SetStringAnalysisInput('SurfaceIntersection', 'STEPFileName', [out])
    vsp.SetIntAnalysisInput('SurfaceIntersection', 'STEPRepresentation', [vsp.STEP_BREP])
    for n in ('CURVFileFlag', 'IGESFileFlag', 'P3DFileFlag', 'SRFFileFlag', 'XYZIntCurveFlag'):
        try:
            vsp.SetIntAnalysisInput('SurfaceIntersection', n, [0])
        except Exception:
            pass
    if iges:
        vsp.SetIntAnalysisInput('SurfaceIntersection', 'IGESFileFlag', [1])
        vsp.SetStringAnalysisInput('SurfaceIntersection', 'IGESFileName', [iges])
    vsp.ExecAnalysis('SurfaceIntersection')
    while errs.GetNumTotalErrors():
        print('VSP', errs.PopLastError().GetErrorString())
    return os.path.exists(out) and os.path.getsize(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('case', choices=sorted(CASES))
    ap.add_argument('--ih', type=float, default=0.0)
    ap.add_argument('--out', default=None)
    ap.add_argument('--common', default=None)
    a = ap.parse_args()
    ihtag = ('%g' % a.ih).replace('-', 'm').replace('.', 'p')
    out = a.out or os.path.join(ROOT, 'geom_v2', f'{a.case}_iH{ihtag}')
    common = a.common or os.path.join(ROOT, 'geom_v2', 'common')
    os.makedirs(out, exist_ok=True); os.makedirs(common, exist_ok=True)

    vsp.ClearVSPModel(); vsp.ReadVSPFile(SRC); vsp.Update()
    P = find_geoms()
    secs = wing_sections(a.case)
    set_wing(P['wing'], secs)
    set_tail_foils(P['htail']); set_tail_foils(P['vtail'])
    c_ht = set_ih(P['htail'], a.ih)

    info = dict(case=a.case, iH_deg=a.ih, wing_origin=[gp(P['wing'], 'X_Rel_Location'), 0.0, gp(P['wing'], 'Z_Rel_Location')],
                sections=[dict(y=s[0], chord=s[1], twist=s[2], foil=s[3]) for s in secs])
    info['planform'] = planform_metrics(secs, info['wing_origin'][0])
    info['vsp_total'] = {k: gp(P['wing'], k, 'WingGeom') for k in ('TotalSpan', 'TotalArea', 'TotalProjectedSpan', 'TotalChord')}
    # nacelle axis (end-cap centres, global) + radii, used by the duct builder
    N = P['nacelle']
    def ctr(u):
        ps = [pt(N, u, w) for w in (0.0, 0.25, 0.5, 0.75)]
        return [sum(p[i] for p in ps) / 4 for i in range(3)], max(abs(ps[0][1] - ps[2][1]), abs(ps[1][2] - ps[3][2])) / 2
    (c0, r0), (c1, r1) = ctr(0.0), ctr(1.0)
    info['nacelle'] = dict(inlet_center=c0, exhaust_center=c1, inlet_radius=r0, exhaust_radius=r1)
    # HT 25 % root-chord point check (must equal the iH=0 value 23.18914+0.25*c, 0.49664, 0.9525)
    le = pt(P['htail'], 0.0, 0.5); te = [(a_ + b_) / 2 for a_, b_ in zip(pt(P['htail'], 0.0, 0.0), pt(P['htail'], 0.0, 1.0))]
    info['htail'] = dict(root_chord=c_ht, p25_root=[le[i] + 0.25 * (te[i] - le[i]) for i in range(3)],
                         p25_expected=[23.189137550299908 + 0.25 * c_ht, 0.4966389776357825, 0.9525])
    # end sections of the wing: chord vector pitch angle (3D angle to x axis) for the twist check
    def pitch(g, u):
        le = pt(g, u, 0.5); te = [(a_ + b_) / 2 for a_, b_ in zip(pt(g, u, 0.0), pt(g, u, 1.0))]
        v = [te[i] - le[i] for i in range(3)]
        return -math.copysign(1.0, v[2]) * math.degrees(math.acos(v[0] / math.sqrt(sum(t * t for t in v)))), math.sqrt(sum(t * t for t in v)), le, te
    info['wing_root_tip'] = dict(root=pitch(P['wing'], 0.0)[:2], tip=pitch(P['wing'], 1.0)[:2],
                                 root_le=pitch(P['wing'], 0.0)[2], tip_le=pitch(P['wing'], 1.0)[2])

    vsp.WriteVSPFile(os.path.join(out, 'case.vsp3'), vsp.SET_ALL)
    r = {}
    r['wing'] = export_step(P, 'wing', os.path.join(out, f'{a.case}_wing.stp'), iges=os.path.join(out, f'{a.case}_wing.igs'))
    r['htail'] = export_step(P, 'htail', os.path.join(common, f'htail_iH{ihtag}.stp')) if not os.path.exists(os.path.join(common, f'htail_iH{ihtag}.stp')) else 'kept'
    for k in ('vtail', 'nacelle', 'pylon'):      # case independent: exported once, shared by all cases
        f = os.path.join(common, f'{k}.stp')
        r[k] = export_step(P, k, f) if not os.path.exists(f) else 'kept'
    info['exports'] = r
    json.dump(info, open(os.path.join(out, 'vsp_info.json'), 'w'), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == '__main__':
    main()
