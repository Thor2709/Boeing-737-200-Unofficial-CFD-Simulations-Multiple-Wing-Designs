import csv
import json
import shutil
from types import SimpleNamespace

from b737wing.cfd.solve import transient
from b737wing.cfd.solve.transient import FluentTransientBackend, run_transient


def _quiet(*args, **kwargs):
    return None


class FakeBackend:
    def __init__(self, audit_ok=True, on_advance=None):
        self.audit_ok = audit_ok
        self.on_advance = on_advance
        self.calls = []
        self.advances = []
        self.exports = []
        self.applied = {}
        self.not_set = {}
        self.t = 0
        self.saved = None

    def read(self, case, data):
        self.calls.append("read")

    def audit(self):
        return self.audit_ok

    def set_transient(self, dt, inner, chunk, schemes=()):
        self.schemes = schemes
        self.calls.append("set_transient")
        return {}

    def setup_monitors(self, out, every):
        pass

    def setup_live_views(self, out, every):
        self.calls.append("live")

    def advance(self, n, inner):
        self.advances.append(n)
        self.t += n
        if self.on_advance:
            self.on_advance(self)

    def sample(self):
        return {"CL": self.t / 100.0, "CD": 0.1, "CM": -0.2}

    def flow_time(self):
        return self.t * 2.5e-4

    def export_ensight(self, path):
        self.exports.append(path.name)

    def save_final(self, path):
        self.saved = path


def _settings(tmp_path, **kw):
    s = {"case": "C5", "init_case": "a.cas.h5", "init_data": "a.dat.h5",
         "out": str(tmp_path), "dt": 2.5e-4, "steps": 100, "inner": 20,
         "chunk": 10, "autosave_every": 50, "export_every": 0, "anim_every": 0,
         "max_hours": 6.0}
    s.update(kw)
    return s


def _result(tmp_path):
    return json.loads((tmp_path / "transient_result.json").read_text())


def test_dry_defaults(capsys):
    rc = transient.main(["--case", "C5", "--init-case", "a", "--init-data", "b",
                         "--out", "x", "--dry"])
    data = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert (data["dt"], data["steps"], data["inner"], data["chunk"]) == (
        1.0e-4, 200, 15, 10)
    assert (data["urf_profile"], data["steady_iters"]) == ("default", 100)


def test_chunked_progress_and_stats(tmp_path):
    fb = FakeBackend()
    res = run_transient(fb, _settings(tmp_path), log=_quiet)
    assert res["status"] == "completed"
    assert fb.advances == [10] * 10
    with (tmp_path / "progress.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 10 and rows[-1]["step"] == "100"
    cl = _result(tmp_path)["stats"]["CL"]
    assert cl["n"] == 10 and cl["min"] == 0.1 and cl["max"] == 1.0
    assert abs(cl["mean"] - 0.55) < 1e-12
    assert fb.saved is not None


def test_stop_file(tmp_path):
    def stop(_backend):
        (tmp_path / "STOP").write_text("x")
    fb = FakeBackend(on_advance=stop)
    res = run_transient(fb, _settings(tmp_path), log=_quiet)
    assert res["status"] == "stopped_by_owner"
    assert fb.advances == [10] and fb.saved is not None


def test_budget_reached(tmp_path):
    ticks = iter(range(0, 100000, 4000))
    fb = FakeBackend()
    res = run_transient(fb, _settings(tmp_path, max_hours=1.0),
                        clock=lambda: float(next(ticks)), log=_quiet)
    assert res["status"] == "budget_reached"
    assert fb.saved is not None


def test_audit_failure(tmp_path):
    fb = FakeBackend(audit_ok=False)
    res = run_transient(fb, _settings(tmp_path))
    assert res["status"] == "audit_failed"
    assert fb.advances == [] and "set_transient" not in fb.calls
    assert _result(tmp_path)["status"] == "audit_failed"


class _Solver:
    """Solver node whose `time` setter rejects values outside an allowed set."""

    def __init__(self, allowed):
        object.__setattr__(self, "_allowed", allowed)
        object.__setattr__(self, "_value", None)

    def __setattr__(self, name, value):
        if name != "time":
            object.__setattr__(self, name, value)
        elif value not in self._allowed:
            raise ValueError(f"{value} not allowed")
        else:
            object.__setattr__(self, "_value", value)

    @property
    def time(self):
        return lambda: self._value


def test_time_scheme_fallback(tmp_path):
    solver = _Solver({"unsteady-2nd-order", "unsteady-1st-order"})
    rc = SimpleNamespace(transient_controls=SimpleNamespace(type=None),
                         parameters=SimpleNamespace())
    settings = SimpleNamespace(
        setup=SimpleNamespace(general=SimpleNamespace(solver=solver)),
        solution=SimpleNamespace(run_calculation=rc))
    backend = FluentTransientBackend(SimpleNamespace(settings=settings), tmp_path)
    info = backend.set_transient(2.5e-4, 20, 10)
    assert info["time_scheme"] == "unsteady-2nd-order"
    assert len(info["time_scheme_rejections"]) == 1
    assert "bounded" in info["time_scheme_rejections"][0]
    assert rc.parameters.time_step_size == 2.5e-4


def test_gui_passes_ui_mode(tmp_path):
    seen = []

    def launcher(**kw):
        seen.append(kw)
        return SimpleNamespace()
    transient.launch_session(tmp_path, True, launcher)
    transient.launch_session(tmp_path, False, launcher)
    assert seen[0]["ui_mode"] == "gui" and seen[0]["gpu"] == [0]
    assert seen[0]["precision"] == "single" and seen[0]["processor_count"] == 1
    assert "ui_mode" not in seen[1]


def test_export_cadence(tmp_path):
    fb = FakeBackend()
    run_transient(fb, _settings(tmp_path, export_every=20), log=_quiet)
    assert fb.exports == [f"frame_{s:05d}" for s in (20, 40, 60, 80, 100)]


def test_share_ensight_geometry(tmp_path):
    from types import SimpleNamespace
    from b737wing.cfd.solve.transient import share_ensight_geometry
    owner = SimpleNamespace()
    for step in (2, 4):
        frame = tmp_path / f"frame_{step:05d}"
        frame.with_suffix(".geo").write_text("g")
        frame.with_suffix(".encas").write_text(f'model: "frame_{step:05d}.geo"\n')
        share_ensight_geometry(frame, owner)
    assert (tmp_path / "frame_00002.geo").exists()
    assert not (tmp_path / "frame_00004.geo").exists()
    assert '"frame_00002.geo"' in (tmp_path / "frame_00004.encas").read_text()


def test_divergence_stops_run(tmp_path):
    fb = FakeBackend()
    fb.sample = lambda: {"CL": float("nan"), "CD": 0.1, "CM": -0.2}
    res = run_transient(fb, _settings(tmp_path), log=_quiet)
    assert res["status"] == "diverged"
    assert fb.advances == [10]

def test_stalled_chunk_counts_as_divergence(tmp_path):
    backend = FakeBackend()
    original = backend.advance

    def advance(n, inner):
        if len(backend.advances) >= 2:
            backend.advances.append((n, inner))
            return
        original(n, inner)

    backend.advance = advance
    result = run_transient(backend, _settings(tmp_path, steps=100, chunk=10),
                           clock=lambda: 0.0, log=lambda *a, **k: None)
    assert result["status"] == "diverged"
    assert "stopped the chunk" in result["message"]


def test_missing_stability_control_aborts(tmp_path):
    backend = FakeBackend()

    def set_controls(max_temperature, urf_profile="default"):
        backend.not_set = {"max_temperature": "not_set: inactive"}

    backend.set_controls = set_controls
    result = run_transient(backend, _settings(tmp_path),
                           clock=lambda: 0.0, log=lambda *a, **k: None)
    assert "stability controls not applied: max_temperature" in result["message"]
    assert backend.advances == []


def test_negative_drag_counts_as_divergence(tmp_path):
    backend = FakeBackend()
    backend.sample = lambda: {"CL": 1.05, "CD": -0.43, "CM": 0.06}
    result = run_transient(backend, _settings(tmp_path),
                           clock=lambda: 0.0, log=lambda *a, **k: None)
    assert result["status"] == "diverged"


def test_steady_iterations_run_before_transient(tmp_path):
    backend = FakeBackend()
    order = []
    backend.steady_iterate = lambda n: order.append(("steady", n))
    original = backend.set_transient

    def set_transient(*a, **k):
        order.append(("transient",))
        return original(*a, **k)

    backend.set_transient = set_transient
    run_transient(backend, _settings(tmp_path, steps=10, chunk=10, steady_iters=50),
                  clock=lambda: 0.0, log=lambda *a, **k: None)
    assert order[:2] == [("steady", 50), ("transient",)]


def test_export_start_skips_spin_up(tmp_path):
    backend = FakeBackend()
    run_transient(backend, _settings(tmp_path, steps=40, chunk=10, export_every=10,
                                     export_start=30),
                  clock=lambda: 0.0, log=lambda *a, **k: None)
    assert backend.exports == ["frame_00030", "frame_00040"]


def test_disk_guard_stops_exports(tmp_path, monkeypatch):
    backend = FakeBackend()
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(transient.shutil, "disk_usage",
                        lambda p: usage._replace(free=int(5e9)))
    result = run_transient(backend, _settings(tmp_path, steps=20, chunk=10,
                                              export_every=10, min_free_gb=15.0),
                           clock=lambda: 0.0, log=lambda *a, **k: None)
    assert backend.exports == []
    assert "5.0 GB free" in result["disk_guard"]
    assert result["status"] == "completed"


def test_sbes_not_applied_aborts(tmp_path):
    backend = FakeBackend()

    def set_turbulence(model, momentum):
        backend.not_set = {"sbes": "not_set: rejected"}

    backend.set_turbulence = set_turbulence
    result = run_transient(backend, _settings(tmp_path, turbulence="sbes"),
                           clock=lambda: 0.0, log=lambda *a, **k: None)
    assert "stability controls not applied: sbes" in result["message"]
    assert backend.advances == []
