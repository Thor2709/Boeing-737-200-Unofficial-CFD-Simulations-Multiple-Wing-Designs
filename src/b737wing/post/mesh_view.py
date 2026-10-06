"""Render the aircraft walls from a Fluent .msh.h5 (no Fluent licence needed).

python -m b737wing.post.mesh_view <mesh.msh.h5> <out_prefix>
Reads boundary faces with h5py, mirrors the half model about y = 0, colours by wall zone,
writes <out_prefix>_iso.png, _top.png, _front.png, _nacelle.png (pyvista off-screen).
"""
import sys
import h5py
import numpy as np
import pyvista as pv
from b737wing.cfd.solve.fluent_io import resolve_zone_groups

pv.OFF_SCREEN = True
WALLS = {"fuselage": "#c8ccd2", "mainwing": "#3b82c4", "horstab": "#4fa3e0", "vertstab": "#4fa3e0",
         "engine": "#d98c2b", "nacelle_duct": "#b5441f", "pylon": "#8d6e63", "fairing": "#59a14f"}


def read_walls(path, mirror=True, return_groups=False):
    with h5py.File(path, "r") as source:
        mesh = source["meshes/1"]
        coords = np.vstack([mesh["nodes/coords"][key][()]
                            for key in sorted(mesh["nodes/coords"], key=int)])
        topology = mesh["faces/zoneTopology"]
        names = topology["name"][0].decode().split(";")
        groups = resolve_zone_groups(names)
        wall_names = set(groups["total"])
        face_nodes = mesh["faces/nodes"]
        sections = {int(face_nodes[key].attrs["minId"][0]): face_nodes[key]
                    for key in face_nodes}
        walls = {}
        for name, lower_id in zip(names, topology["minId"][()]):
            if name not in wall_names:
                continue
            section = sections[int(lower_id)]
            node_counts = section["nnodes"][()]
            nodes = section["nodes"][()] - 1
            cells = np.insert(nodes, np.concatenate([[0], np.cumsum(node_counts)[:-1]]), node_counts)
            poly = pv.PolyData(coords, cells).clean()
            walls[name] = (poly.merge(poly.reflect((0, 1, 0), point=(0, 0, 0)))
                           if mirror else poly)
    return (walls, groups) if return_groups else walls


def render(walls, prefix, groups=None):
    component_by_zone = {}
    if groups is not None:
        for component in ("wing", "fuselage", "htail", "vtail", "nacelle", "duct", "fairing"):
            component_by_zone.update({name: component for name in groups[component]})
    component_colors = {"wing": WALLS["mainwing"], "fuselage": WALLS["fuselage"],
                        "htail": WALLS["horstab"], "vtail": WALLS["vertstab"],
                        "nacelle": WALLS["engine"], "duct": WALLS["nacelle_duct"],
                        "fairing": WALLS["fairing"]}

    def color_for(name):
        return WALLS.get(name, component_colors.get(component_by_zone.get(name), "#c8ccd2"))

    views = {"iso": dict(pos=(-25, -45, 22), up=(0, 0, 1)), "top": dict(pos=(15, 0, 80), up=(1, 0, 0)),
             "front": dict(pos=(-60, 0, 1), up=(0, 0, 1))}
    for tag, v in views.items():
        p = pv.Plotter(window_size=(1920, 1080)); p.set_background("white"); p.enable_anti_aliasing("ssaa")
        for name, mesh in walls.items():
            p.add_mesh(mesh, color=color_for(name), smooth_shading=True, specular=0.3, label=name)
        p.add_legend(bcolor="white", size=(0.14, 0.22))
        p.camera_position = [v["pos"], (14.5, 0, 0), v["up"]]; p.reset_camera(); p.camera.zoom(1.25)
        p.screenshot(f"{prefix}_{tag}.png"); p.close()
    p = pv.Plotter(window_size=(1600, 1000)); p.set_background("white"); p.enable_anti_aliasing("ssaa")
    if groups is None:
        nacelle_zones = [name for name in ("engine", "nacelle_duct", "pylon", "mainwing", "fairing", "fuselage")
                         if name in walls]
    else:
        nacelle_zones = (groups["nacelle"] + groups["duct"] + groups["wing"]
                         + groups["fairing"] + groups["fuselage"])
    for name in nacelle_zones:
        p.add_mesh(walls[name], color=color_for(name), smooth_shading=True,
                   show_edges=name in ("engine", "nacelle_duct"), edge_color="#555555")
    p.camera_position = [(4, 9, -3.5), (11.5, 5.1, -1.0), (0, 0, 1)]; p.camera.zoom(1.0)
    p.screenshot(f"{prefix}_nacelle.png"); p.close()


if __name__ == "__main__":
    walls, groups = read_walls(sys.argv[1], return_groups=True)
    render(walls, sys.argv[2], groups)
