"""P3 step 2 library: Discovery (PyAnsys Geometry, Python 3.12) builders for the half-aircraft fluid domain.

Used by build_domain.py (and exec'd in the persistent development session). Namespace needs `m` (Modeler).
Frame: x streamwise (aft), y span (half model y >= 0), z up, metres.
"""
import json, os, time, math, uuid
from pathlib import Path
import numpy as np
from ansys.geometry.core.math import Point2D, Point3D, Plane, UNITVECTOR3D_X, UNITVECTOR3D_Y, UNITVECTOR3D_Z, UnitVector3D
from ansys.geometry.core.sketch import Sketch
from ansys.geometry.core.sketch.face import SketchFace
from ansys.geometry.core.misc import UNITS, Distance
from ansys.geometry.core.misc.options import TessellationOptions

HERE = globals().get('HERE') or os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')

# ----------------------------------------------------------------------------- fixed design parameters
BORE_FRAC = 0.85            # duct bore radius = 0.85 x inlet radius (nacelle STEP inlet radius 0.53721 m)
EXH_WALL = 0.035            # minimum wall thickness at the (shortened) exhaust lip, m
FAR = dict(x0=-170.0, x1=370.0, y1=170.0, z0=-170.0, z1=170.0)   # 170 m up/side/down, 340 m aft of the tail (x_tail ~ 29.5-30)
# belly fairing (identical for all cases): lofted ellipses, centre z, a = half width, b = half height
BELLY = dict(x0=8.0, x1=15.0, n=9, zc=-0.55, a0=1.60, da=0.50, b0=0.40, db=0.30)


def retry(fn, *a, **k):
    """PyAnsys<->Discovery 26R1 intermittently answers 'Write calls only allowed inside a write block'; safe to repeat."""
    for t in range(12):
        try:
            return fn(*a, **k)
        except Exception as ex:
            if 'write block' not in str(ex):
                raise
            time.sleep(1.5)
    raise RuntimeError('retry exhausted')


def f(v):
    return float(v.m if hasattr(v, 'm') else v)


def bbox(b):
    x = b.bounding_box
    return [round(f(v), 4) for v in (x.min_corner.x, x.min_corner.y, x.min_corner.z, x.max_corner.x, x.max_corner.y, x.max_corner.z)]


def all_bodies(design):
    out = list(design.bodies)
    def walk(c):
        out.extend(c.bodies)
        for cc in c.components:
            walk(cc)
    for c in design.components:
        walk(c)
    return out


# ----------------------------------------------------------------------------- primitives
def section_sketch(s):
    """Ellipse/circle section of width w (y), height h (z) in the plane x = s['x'] centred at (x, y, z)."""
    a, b = s['w'] / 2, s['h'] / 2
    c = Point3D([s['x'], s['y'], s['z']], UNITS.m)
    if abs(a - b) < 1e-9:
        sk = Sketch(Plane(c, direction_x=UNITVECTOR3D_Y, direction_y=UNITVECTOR3D_Z)); sk.circle(Point2D([0, 0], UNITS.m), a * UNITS.m)
    elif b > a:   # tall ellipse: major axis along global Z (sketch x = Z)
        sk = Sketch(Plane(c, direction_x=UNITVECTOR3D_Z, direction_y=UNITVECTOR3D_Y)); sk.ellipse(Point2D([0, 0], UNITS.m), b * UNITS.m, a * UNITS.m)
    else:         # wide ellipse: major axis along global Y
        sk = Sketch(Plane(c, direction_x=UNITVECTOR3D_Y, direction_y=UNITVECTOR3D_Z)); sk.ellipse(Point2D([0, 0], UNITS.m), a * UNITS.m, b * UNITS.m)
    return sk


def loft_solid(design, name, secs):
    """Loft of ellipse sections + the two end caps, stitched into one solid."""
    comp = retry(design.add_component, name)
    helpers = []; profiles = []
    for i, s in enumerate(secs):
        h = retry(comp.create_surface, f'sec{i}', section_sketch(s)); helpers.append(h); profiles.append([e.shape for e in h.edges])
    body = retry(comp.create_body_from_loft_profile, name, profiles)
    for h in helpers:
        retry(comp.delete_body, h)
    caps = [retry(comp.create_surface, f'cap{k}', section_sketch(s)) for k, s in enumerate((secs[0], secs[-1]))]
    retry(m.repair_tools.find_and_fix_stitch_faces, [body] + caps, max_distance=1e-4)
    return comp


def make_fuselage(design):
    V = json.load(open(os.path.join(DATA, 'vsp_data.json'))); st = V['parts']['fuselage']['stations']
    dense = json.load(open(os.path.join(DATA, 'fuselage_dense.json')))['sections']
    ends = lambda s: dict(x=s['x'], y=s['y'], z=s['z'], w=s['w'], h=s['h'])
    secs = [ends(st[0])] + [ends(s) for s in dense] + [ends(st[-1])]
    return loft_solid(design, 'Fuselage', secs)


def make_belly(design):
    B = BELLY
    secs = []
    for i in range(B['n']):
        s = i / (B['n'] - 1); e = math.sin(math.pi * s)
        secs.append(dict(x=B['x0'] + (B['x1'] - B['x0']) * s, y=0.0, z=B['zc'], w=2 * (B['a0'] + B['da'] * e), h=2 * (B['b0'] + B['db'] * e)))
    return loft_solid(design, 'BellyFairing', secs)


def make_pylon(design, side_idx=0, name='Pylon'):
    """Rebuild of the engine-to-wing fairing (pylon) by a loft of VSP-sampled sections (VSP STEP sheets are open)."""
    FS = json.load(open(os.path.join(DATA, 'fairing_sections.json')))[str(side_idx)]
    comp = retry(design.add_component, name)
    us = [s for s in FS if 1 / 6 - 1e-9 <= s['u'] <= 5 / 6 + 1e-9]
    helpers = []; profiles = []
    for k, s in enumerate(us):
        pts = np.array(s['pts']); c = pts.mean(0); _, _, vt = np.linalg.svd(pts - c); ex = vt[0]; ey = np.cross(vt[2], ex)
        p2 = np.c_[(pts - c) @ ex, (pts - c) @ ey]; iTE = int(np.argmax(p2[:, 0]))
        sk = Sketch(Plane(Point3D(list(c), UNITS.m), direction_x=UnitVector3D(list(ex)), direction_y=UnitVector3D(list(ey))))
        P2 = lambda a: [Point2D([float(q[0]), float(q[1])], UNITS.m) for q in a]
        sk.nurbs_from_2d_points(P2(p2[:iTE + 1])); sk.nurbs_from_2d_points(P2(p2[iTE:]))
        fc = SketchFace(); fc._edges = list(sk.edges); sk.face(fc)
        h = retry(comp.create_surface, f'sec{k}', sk); helpers.append(h); profiles.append([e.shape for e in h.edges])
    body = retry(comp.create_body_from_loft_profile, 'Pylon', profiles)
    for h in helpers[1:-1]:
        retry(comp.delete_body, h)
    retry(m.repair_tools.find_and_fix_stitch_faces, [body, helpers[0], helpers[-1]], max_distance=1e-3)
    return comp


def box_body(design, name, x0, x1, y0, y1, z0, z1):
    comp = retry(design.add_component, name)
    sk = Sketch(Plane(Point3D([x0, 0, 0], UNITS.m), direction_x=UNITVECTOR3D_Y, direction_y=UNITVECTOR3D_Z))
    sk.box(Point2D([(y0 + y1) / 2, (z0 + z1) / 2], UNITS.m), (y1 - y0) * UNITS.m, (z1 - z0) * UNITS.m)
    b = retry(comp.extrude_sketch, name, sk, Distance(x1 - x0, UNITS.m))
    return comp, b


def axis_frame(d):
    d = np.asarray(d, float); d /= np.linalg.norm(d)
    ex = np.cross([0, 1, 0], d); ex /= np.linalg.norm(ex); ey = np.cross(d, ex)
    return d, ex, ey


def cyl_body(design, name, p0, d, r, length):
    """Cylinder of radius r, starting at p0 along unit direction d."""
    d, ex, ey = axis_frame(d)
    comp = retry(design.add_component, name)
    sk = Sketch(Plane(Point3D(list(p0), UNITS.m), direction_x=UnitVector3D(list(ex)), direction_y=UnitVector3D(list(ey))))
    sk.circle(Point2D([0, 0], UNITS.m), r * UNITS.m)
    b = retry(comp.extrude_sketch, name, sk, Distance(length, UNITS.m))
    return comp, b


# ----------------------------------------------------------------------------- tessellation helpers
def tess_cloud(body, dev=0.002, maxedge=0.15):
    """Surface sample cloud of a body: triangle vertices + centroids, as an (n,3) array in metres."""
    t = body.get_raw_tessellation(TessellationOptions(surface_deviation=dev, angle_deviation=0.1, max_aspect_ratio=20, max_edge_length=maxedge), reset_cache=True)
    return t


def tess_arrays(t):
    """raw tessellation dict -> {face_key: (verts (n,3), tris (k,3))}"""
    out = {}
    for k, v in t.items():
        if not isinstance(v, dict) or not len(v.get('vertices', [])):
            continue
        V = np.asarray(v['vertices'], dtype=float).reshape(-1, 3)
        fa = np.asarray(v['faces'], dtype=np.int64)
        T = fa.reshape(-1, 4)[:, 1:] if (len(fa) % 4 == 0 and (fa.reshape(-1, 4)[:, 0] == 3).all()) else fa.reshape(-1, 3)
        out[k] = (V, T)
    return out


# ----------------------------------------------------------------------------- robust building
def sweep_leftovers(design, comp_name=None):
    """delete-body of helper sheets is intermittently lost by PyAnsys: sweep leftovers named sec*/cap*."""
    for c in list(design.components):
        if comp_name and c.name != comp_name:
            continue
        for b in list(c.bodies):
            if b.is_surface and (b.name.startswith('sec') or b.name.startswith('cap')):
                try:
                    retry(c.delete_body, b)
                except Exception as ex:
                    print('leftover delete failed', b.name, str(ex)[:60])
    design._update_design_inplace()


def solid_of(design, name):
    cs = [c for c in design.components if c.name == name]
    if not cs:
        return None
    sol = [b for b in cs[0].bodies if not b.is_surface]
    return sol[0] if len(sol) == 1 and len(cs[0].bodies) == 1 else None


def make_checked(design, fn, name, tries=6):
    for k in range(tries):
        fn(design); design._update_design_inplace(); sweep_leftovers(design, name)
        b = solid_of(design, name)
        if b is not None:
            return b
        print(f'{name}: not a single solid (attempt {k}):', [(x.name, x.is_surface, len(x.faces)) for c in design.components if c.name == name for x in c.bodies], flush=True)
        for c in list(design.components):
            if c.name == name:
                retry(design.delete_component, c)
        design._update_design_inplace()
    raise RuntimeError(f'{name} never became a solid')


# ----------------------------------------------------------------------------- assembly
def insert_step(design, path, name):
    c = retry(design.insert_file, path); c.set_name(name)
    return c


def keep_side(comp, sign=+1):
    """Components imported from VSP contain both halves as separate solids: keep the y>0 one (sign=+1)."""
    bs = list(comp.bodies)
    if len(bs) == 1:
        return bs[0]
    mid = [(f(b.bounding_box.min_corner.y) + f(b.bounding_box.max_corner.y)) / 2 for b in bs]
    keep = max(range(len(bs)), key=lambda i: sign * mid[i])
    for i, b in enumerate(bs):
        if i != keep:
            retry(comp.delete_body, b)
    return bs[keep]


def cloud_of(body):
    """(n,3) surface sample cloud (vertices + triangle centroids) of a body."""
    A = tess_arrays(tess_cloud(body))
    pts = []
    for V, T in A.values():
        pts.append(V); pts.append(V[T].mean(1))
    return np.vstack(pts)


def nacelle_cut_s(cloud, c0, d, rb):
    """Axial station s (from the inlet centre) where the outer radius has shrunk to bore + EXH_WALL (shortened exhaust)."""
    P = cloud - np.asarray(c0)
    s = P @ d
    for sv in np.arange(4.0, 5.3, 0.005):
        sel = np.abs(s - sv) < 0.02
        if sel.sum() < 3:
            continue
        r = np.abs(P[sel, 1]).max()       # horizontal half width (the axis is parallel to x, sections are circles here)
        if r <= rb + EXH_WALL:
            return float(sv), float(r)
    raise RuntimeError('no exhaust station found')


def build_aircraft(design, wing_stp, htail_stp, common, info, log=print, with_belly=True):
    """Build all components, record surface clouds, make the nacelle flow-through, unite to ONE solid.
    Returns (aircraft_body, clouds dict, notes dict)."""
    t0 = time.time(); notes = {}; clouds = {}
    tick = lambda s: log(f'[{round(time.time() - t0)}s] {s}')
    fus = make_checked(design, make_fuselage, 'Fuselage'); tick(f'fuselage faces {len(fus.faces)} vol {f(fus.volume):.3f}')
    clouds['fuselage'] = cloud_of(fus)
    # components from VSP STEP
    comps = {}
    for key, path in (('wing', wing_stp), ('htail', htail_stp), ('vtail', os.path.join(common, 'vtail.stp')), ('nacelle', os.path.join(common, 'nacelle.stp'))):
        comps[key] = insert_step(design, path, key); tick(f'imported {key}')
    bodies = {k: keep_side(c, +1) for k, c in comps.items() if k in ('htail', 'nacelle')}
    bodies['wing'] = comps['wing'].bodies[0]; bodies['vtail'] = comps['vtail'].bodies[0]
    for k in ('wing', 'vtail'):
        assert len(comps[k].bodies) == 1 and not bodies[k].is_surface, f'{k}: expected one solid'
    design._update_design_inplace()
    # pylon (+y side)
    pyl = make_checked(design, lambda d: make_pylon(d, 0, 'Pylon'), 'Pylon'); tick(f'pylon faces {len(pyl.faces)} vol {f(pyl.volume):.4f} bbox {bbox(pyl)}')
    # belly fairing (optional)
    belly = None
    try:
        if not with_belly:
            raise RuntimeError('belly fairing disabled')
        belly = make_checked(design, make_belly, 'BellyFairing'); tick(f'belly fairing faces {len(belly.faces)} vol {f(belly.volume):.3f}')
        clouds['fairing'] = cloud_of(belly)
    except Exception as ex:
        notes['fairing'] = 'FAILED to build: ' + str(ex)[:150]
    # nacelle axis from VSP (inlet/exhaust end-cap centres)
    N = info['nacelle']; c0 = np.array(N['inlet_center']); c1 = np.array(N['exhaust_center'])
    d = (c1 - c0) / np.linalg.norm(c1 - c0)
    rin = N['inlet_radius']; rb = BORE_FRAC * rin
    nac = bodies['nacelle']; ncl = cloud_of(nac)
    s_cut, r_cut = nacelle_cut_s(ncl, c0, d, rb)
    notes['duct'] = dict(bore_radius=rb, inlet_radius=rin, exhaust_radius_vsp=N['exhaust_radius'], cut_station_from_inlet=s_cut,
                         outer_radius_at_cut=r_cut, nacelle_length_vsp=float(np.linalg.norm(c1 - c0)), axis_dir=d.tolist(), inlet_center=c0.tolist())
    tick(f'nacelle: bore r={rb:.4f}, exhaust shortened to s={s_cut:.3f} m (outer r {r_cut:.4f}; VSP length {np.linalg.norm(c1 - c0):.3f})')
    tc, tool = cyl_body(design, 'cut_exhaust', c0 + d * s_cut, d, 1.5, 3.0)
    retry(nac.subtract, tool); design._update_design_inplace()
    clouds['engine'] = cloud_of(nac)
    clouds['pylon'] = cloud_of(pyl)
    retry(nac.unite, pyl); design._update_design_inplace(); tick('nacelle + pylon united')
    bc, bore = cyl_body(design, 'bore', c0 - d * 0.6, d, rb, s_cut + 1.2)
    retry(nac.subtract, bore); design._update_design_inplace(); tick(f'bore subtracted: nacelle faces {len(nac.faces)} vol {f(nac.volume):.4f}')
    clouds['comp_ids'] = None
    for key in ('wing', 'htail', 'vtail'):
        clouds[{'wing': 'mainwing', 'htail': 'horstab', 'vtail': 'vertstab'}[key]] = cloud_of(bodies[key])
    # unite everything into the fuselage solid
    order = ['htail', 'vtail', 'wing'] + (['fairing'] if belly is not None else []) + ['nacelle']
    objs = {'htail': bodies['htail'], 'vtail': bodies['vtail'], 'wing': bodies['wing'], 'nacelle': nac, 'fairing': belly}
    for k in order:
        retry(fus.unite, [objs[k]]); design._update_design_inplace(); tick(f'united {k}: faces {len(fus.faces)}')
    return fus, clouds, notes


# ----------------------------------------------------------------------------- checks
def run_checks(bodies, log=print):
    """Discovery repair-tool checks; returns {name: count or error}."""
    res = {}
    for fn in ('inspect_geometry', 'find_interferences', 'find_missing_faces', 'find_duplicate_faces', 'find_small_faces', 'find_split_edges', 'find_short_edges', 'find_extra_edges', 'find_inexact_edges', 'find_stitch_faces', 'find_sliver_faces'):
        if not hasattr(m.repair_tools, fn):
            if fn in ('inspect_geometry', 'find_interferences', 'find_missing_faces', 'find_duplicate_faces'):
                raise RuntimeError(f'required geometry check API is unavailable: {fn}')
            continue
        try:
            r = getattr(m.repair_tools, fn)(bodies)
            if fn == 'inspect_geometry':
                reports = [r] if hasattr(r, 'issues') else list(r) if r is not None else None
                if reports is None or any(x is None or not hasattr(x, 'issues') or x.issues is None for x in reports):
                    res[fn] = None
                else:
                    res[fn] = sum(len(x.issues) for x in reports)
            else:
                res[fn] = check_result_issue_count(r)
        except Exception as ex:
            res[fn] = 'ERR ' + str(ex)[:100]
    log('checks ' + json.dumps(res))
    return res


# ----------------------------------------------------------------------------- domain + labels
def make_domain(design, aircraft, log=print):
    """Half box (y >= 0) minus the aircraft solid."""
    bc, box = box_body(design, 'Air', FAR['x0'], FAR['x1'], 0.0, FAR['y1'], FAR['z0'], FAR['z1'])
    log(f'box faces {len(box.faces)} vol {f(box.volume):.1f}')
    retry(box.subtract, [aircraft]); design._update_design_inplace()
    return box


def match_faces(faces, T):
    """Raw-tessellation keys do not equal Face.id: match by area and centroid."""
    info = {}
    for k, (V, Tr) in T.items():
        a = 0.5 * np.linalg.norm(np.cross(V[Tr[:, 1]] - V[Tr[:, 0]], V[Tr[:, 2]] - V[Tr[:, 0]]), axis=1)
        info[k] = (a.sum(), (V.min(0) + V.max(0)) / 2, V.max(0) - V.min(0))
    keymap = {}; used = set()
    for fc in sorted(faces, key=lambda x: -f(x.area)):
        A = f(fc.area); bb = fc.bounding_box; lo = np.array([f(bb.min_corner.x), f(bb.min_corner.y), f(bb.min_corner.z)]); hi = np.array([f(bb.max_corner.x), f(bb.max_corner.y), f(bb.max_corner.z)]); c = (lo + hi) / 2
        best = min((k for k in info if k not in used), key=lambda k: abs(info[k][0] - A) / max(A, 1e-9) + np.linalg.norm(info[k][1] - c) / (1 + abs(A) ** 0.5))
        keymap[fc.id] = best; used.add(best)
    return keymap


LABELS = ['mainwing', 'horstab', 'vertstab', 'fuselage', 'engine', 'nacelle_duct', 'pylon', 'fairing', 'symmetry', 'farfield']


def classify_faces(domain, clouds, notes, log=print):
    """Return {face_obj: label}; also stats. Uses analytic rules (box planes, bore cylinder) and nearest-surface votes."""
    from scipy.spatial import cKDTree
    D = notes['duct']; c0 = np.array(D['inlet_center']); d = np.array(D['axis_dir']); rb = D['bore_radius']; sc = D['cut_station_from_inlet']
    trees = {k: cKDTree(v) for k, v in clouds.items() if isinstance(v, np.ndarray)}
    names = list(trees)
    T = tess_arrays(domain.get_raw_tessellation(TessellationOptions(surface_deviation=0.005, angle_deviation=0.2, max_aspect_ratio=20, max_edge_length=0.4), reset_cache=True))
    out = []; tol = 1e-4
    keymap = match_faces(domain.faces, T)
    for fc in domain.faces:
        V, Tr = T[keymap[fc.id]]
        cen = V[Tr].mean(1)
        lo, hi = V.min(0), V.max(0)
        lab = None; why = ''
        if abs(lo[1]) < tol and abs(hi[1]) < tol:
            lab = 'symmetry'
        elif any((abs(lo[i] - v) < tol and abs(hi[i] - v) < tol) for i, v in ((0, FAR['x0']), (0, FAR['x1']), (1, FAR['y1']), (2, FAR['z0']), (2, FAR['z1']))):
            lab = 'farfield'
        else:
            P = V - c0; s = P @ d; rad = np.linalg.norm(P - np.outer(s, d), axis=1)
            if abs(rad.max() - rb) < 2e-3 and rad.min() > 0.8 * rb and np.ptp(s) > 1.0 and s.min() > -0.01 and s.max() < sc + 0.01:   # bore cylinder (tessellation vertices may sit inside the radius)
                lab = 'nacelle_duct'
            else:
                dist = np.stack([trees[n].query(cen)[0] for n in names], 1)
                win = dist.argmin(1)
                cnt = np.bincount(win, minlength=len(names))
                lab = names[int(cnt.argmax())]
                md = float(np.median(dist[np.arange(len(win)), win]))
                why = f'votes {dict((names[i], int(c)) for i, c in enumerate(cnt) if c)} med_dist {md:.3f}'
                if md > 0.05:
                    why += ' (UNSURE)'
        out.append((fc, lab, why, f(fc.area)))
    return out


def apply_names(design, L, log=print):
    groups = {}
    for fc, lab, why, a in L:
        groups.setdefault(lab, []).append(fc)
    for lab in LABELS:
        if lab in groups:
            retry(design.create_named_selection, lab, faces=groups[lab])
    log('named selections: ' + ', '.join(f'{k}({len(v)})' for k, v in groups.items()))
    return groups


def require_labels(groups, labels=LABELS):
    missing = [label for label in labels if label not in groups or not groups[label]]
    if missing:
        raise RuntimeError('missing labels ' + str(missing))


def check_result_issue_count(result):
    """Return an explicit check count; None means the geometry API gave no result."""
    if result is None:
        return None
    issues = getattr(result, 'issues', None)
    if issues is not None:
        return len(issues)
    if type(result) is int or isinstance(result, np.integer):
        return int(result)
    try:
        return len(result)
    except TypeError:
        return None


def require_zero_issues(name, result):
    count = check_result_issue_count(result)
    if count is None or count != 0:
        raise RuntimeError(f'{name} did not report zero issues: {count!r}')
    return count


def inspect_geometry_issues(bodies):
    fn = getattr(m.repair_tools, 'inspect_geometry', None)
    if fn is None:
        raise RuntimeError('inspect_geometry API is unavailable')
    result = fn(bodies)
    if result is None:
        raise RuntimeError('inspect_geometry returned no result')
    reports = [result] if hasattr(result, 'issues') else list(result)
    issues = []
    for report in reports:
        if report is None or not hasattr(report, 'issues') or report.issues is None:
            raise RuntimeError('inspect_geometry returned a result without an explicit issues list')
        issues.extend(report.issues)
    return issues


SAVE_TPL = r'''from SpaceClaim.Api.V261 import *
import os
res = []
DST = r"@DST@"
DocumentSave.Execute(DST)
res.append("saved %d B" % os.path.getsize(DST))
try:
    ns = GetRootPart().Document.MainPart.NamedSelections
    res.append("NS: " + ", ".join("%s(%d)" % (n.Name, n.Members.Count) for n in ns))
except Exception as e:
    res.append("NS list failed " + str(e)[:80])
result = {"r": "\n".join(res)}
'''


def save_dsco(path, workdir):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    staged_path = path + f'.{uuid.uuid4().hex}.tmp.dsco'
    sp = os.path.join(workdir, f'save_tmp_{uuid.uuid4().hex}.py')
    try:
        open(sp, 'w').write(SAVE_TPL.replace('@DST@', staged_path.replace('/', chr(92))))
        r = m.run_script_file(sp)
        result = r[0].get('r') if r and r[0] else None
        if not isinstance(result, str) or not result.startswith('saved '):
            raise RuntimeError(f'Discovery did not return an explicit save result: {result!r}')
        if not os.path.isfile(staged_path) or os.path.getsize(staged_path) <= 0:
            raise RuntimeError('Discovery save result did not create a non-empty file for this run')
        os.replace(staged_path, path)
        if not os.path.isfile(path) or os.path.getsize(path) <= 0:
            raise RuntimeError('saved domain file is missing or empty')
        return result
    finally:
        for generated in (sp, staged_path):
            if os.path.exists(generated):
                os.remove(generated)


try:
    from b737wing.config import REPO_ROOT
    ROOT = str(REPO_ROOT)
except ImportError:
    ROOT = str(Path(__file__).resolve().parents[4])
CASE_FILES = lambda case, ih: os.path.join(ROOT, 'geom_v2', f'{case}_iH{ih_tag(ih)}')


def ih_tag(ih):
    return ('%g' % ih).replace('-', 'm').replace('.', 'p')


def dsco_path(case, ih):
    return os.path.join(ROOT, 'cfd_v2', 'geom', f'{case}_domain.dsco' if abs(ih) < 1e-12 else f'{case}_iH{ih_tag(ih)}_domain.dsco')


def tess_stats(domain, groups):
    """min y of the domain, exposed wing semispan, per-label area."""
    T = tess_arrays(domain.get_raw_tessellation(TessellationOptions(surface_deviation=0.005, angle_deviation=0.2, max_aspect_ratio=20, max_edge_length=0.4), reset_cache=True))
    ymin = min(V[:, 1].min() for V, _ in T.values())
    km = match_faces(domain.faces, T)
    st = {}
    for lab, fcs in groups.items():
        pts = np.vstack([T[km[fc.id]][0] for fc in fcs])
        st[lab] = dict(faces=len(fcs), area_m2=round(sum(f(fc.area) for fc in fcs), 4), bbox=[round(float(v), 3) for v in np.r_[pts.min(0), pts.max(0)]])
    # projected (xy) area of the wing faces: planform area of the exposed half wing = half of upper+lower projection
    pa = 0.0
    for fc in groups['mainwing']:
        V, Tr = T[km[fc.id]]
        n = np.cross(V[Tr[:, 1]] - V[Tr[:, 0]], V[Tr[:, 2]] - V[Tr[:, 0]])
        pa += 0.5 * np.abs(n[:, 2]).sum() * 0.5
    return ymin, st, pa


def build_case(case, ih=0.0, log=print):
    t0 = time.time()
    gd = CASE_FILES(case, ih); common = os.path.join(ROOT, 'geom_v2', 'common')
    info = json.load(open(os.path.join(gd, 'vsp_info.json')))
    wing_stp = os.path.join(gd, f'{case}_wing.stp'); ht = os.path.join(common, f'htail_iH{ih_tag(ih)}.stp')
    report = dict(case=case, iH_deg=ih, planform=info['planform'], vsp_total=info['vsp_total'], sections=info['sections'])
    design = m.create_design(f'{case}_iH{ih_tag(ih)}_domain')
    try:
        fus, clouds, notes = build_aircraft(design, wing_stp, ht, common, info, log=log, with_belly=True)
        bs = all_bodies(design)
        if len(bs) != 1:
            raise RuntimeError(f'aircraft is {len(bs)} bodies after union')
        chk_ac = run_checks(bs, log=log)
        for name in ('inspect_geometry', 'find_interferences', 'find_missing_faces', 'find_duplicate_faces'):
            require_zero_issues(f'aircraft {name}', chk_ac.get(name))
        acvol = f(fus.volume); acfaces = len(fus.faces)
        dom = make_domain(design, fus, log=log)
        if len(all_bodies(design)) != 1 or dom.is_surface:
            raise RuntimeError('domain is not a single solid')
        chk_dom = run_checks(all_bodies(design), log=log)
        for name in ('inspect_geometry', 'find_interferences', 'find_missing_faces', 'find_duplicate_faces'):
            require_zero_issues(f'domain {name}', chk_dom.get(name))
        inspection = inspect_geometry_issues(all_bodies(design))
        if inspection:
            raise RuntimeError(f'domain inspect_geometry reported {len(inspection)} issues')
        L = classify_faces(dom, clouds, notes, log=log)
        groups = apply_names(design, L, log=log)
        require_labels(groups)
        ymin, st, pa = tess_stats(dom, groups)
        unsure = [(lab, round(a, 3), why) for _, lab, why, a in L if 'UNSURE' in why]
        report.update(fairing='present', fairing_note=notes.get('fairing', ''), duct=notes['duct'],
                      aircraft_checks=chk_ac, domain_checks=chk_dom, aircraft_volume_m3=acvol, aircraft_faces=acfaces,
                      domain_faces=len(dom.faces), domain_volume_m3=f(dom.volume), labels=st, domain_min_y_tess=float(ymin),
                      wing_exposed_planform_area_half_m2=pa, unsure_faces=unsure,
                      inspect=[dict(type=i.message_type, id=i.message_id, text=i.message) for i in inspection])
        out = dsco_path(case, ih)
        report['dsco'] = out
        report['save'] = save_dsco(out, os.path.join(ROOT, 'geom_v2', '_work'))
        report['seconds'] = round(time.time() - t0)
        json.dump(report, open(os.path.join(gd, 'domain_report.json'), 'w'), indent=1, default=str)
        log(f'CASE {case} DONE in {report["seconds"]} s, fairing present')
        return report
    except Exception as ex:
        import traceback
        log(f'!! {case} geometry build FAILED: {ex}\n{traceback.format_exc()[-800:]}')
        report.setdefault('failures', []).append(dict(error=str(ex)[:300]))
        raise
