from __future__ import annotations

from types import SimpleNamespace

from aibridge_houdini.houdini.inspection import inspect_scene


# ---- fake hou ----------------------------------------------------------


def _fake_node(
    *,
    path: str,
    name: str,
    type_name: str,
    category: str = "Object",
    children: list | None = None,
    parms: dict | None = None,
    parm_tuples: dict | None = None,
    display_node: object | None = None,
    geometry_prim_count: int | None = None,
):
    parms = parms or {}
    parm_tuples = parm_tuples or {}

    def _parm(name: str):
        if name not in parms:
            return None
        return SimpleNamespace(eval=lambda v=parms[name]: v)

    def _parm_tuple(name: str):
        if name not in parm_tuples:
            return None
        return SimpleNamespace(eval=lambda v=parm_tuples[name]: v)

    geo_obj = None
    if geometry_prim_count is not None:
        geo_obj = SimpleNamespace(
            intrinsicValue=lambda key, v=geometry_prim_count: v if key == "primitivecount" else 0
        )

    return SimpleNamespace(
        path=lambda p=path: p,
        name=lambda n=name: n,
        type=lambda: SimpleNamespace(
            name=lambda t=type_name: t,
            category=lambda c=category: SimpleNamespace(name=lambda v=c: v),
        ),
        children=lambda c=children or []: tuple(c),
        parm=_parm,
        parmTuple=_parm_tuple,
        displayNode=lambda d=display_node: d,
        geometry=lambda g=geo_obj: g,
    )


def _fake_hou(*, frame=42.5, fps=24.0, hip="/tmp/test.hip", obj_children=None, selection=None):
    obj_children = obj_children or []
    selection = selection or []
    obj_root = _fake_node(
        path="/obj", name="obj", type_name="manager",
        category="Manager", children=obj_children,
    )
    return SimpleNamespace(
        frame=lambda v=frame: v,
        fps=lambda v=fps: v,
        hipFile=SimpleNamespace(path=lambda v=hip: v),
        node=lambda p, root=obj_root: root if p == "/obj" else None,
        selectedNodes=lambda s=selection: tuple(s),
    )


# ---- top-level structure -----------------------------------------------


def test_inspect_returns_empty_payload_when_hou_missing():
    out = inspect_scene(None)
    assert out["available"] is False
    assert out["error"] == "hou module not available"
    assert out["frame"] is None
    assert out["objects"] == []
    assert out["selected_nodes"] == []
    assert out["cameras"] == []
    assert out["lights"] == []
    assert out["geo_containers"] == []


def test_inspect_returns_required_keys_with_real_shape():
    hou = _fake_hou()
    out = inspect_scene(hou)
    for key in (
        "available", "error", "frame", "fps", "hip_file",
        "objects", "selected_nodes", "cameras", "lights", "geo_containers",
    ):
        assert key in out, key
    assert out["available"] is True
    assert out["frame"] == 42.5
    assert out["fps"] == 24.0
    assert out["hip_file"] == "/tmp/test.hip"


# ---- objects + selection -----------------------------------------------


def test_lists_obj_children():
    geo = _fake_node(path="/obj/geo1", name="geo1", type_name="geo")
    cam = _fake_node(path="/obj/cam1", name="cam1", type_name="cam")
    light = _fake_node(path="/obj/key", name="key", type_name="hlight::2.0")
    hou = _fake_hou(obj_children=[geo, cam, light])

    out = inspect_scene(hou)
    paths = [o["path"] for o in out["objects"]]
    assert paths == ["/obj/geo1", "/obj/cam1", "/obj/key"]


def test_lists_selected_nodes():
    sphere = _fake_node(path="/obj/sphere_geo", name="sphere_geo", type_name="geo")
    hou = _fake_hou(selection=[sphere])
    out = inspect_scene(hou)
    assert [s["path"] for s in out["selected_nodes"]] == ["/obj/sphere_geo"]


# ---- cameras / lights / geo containers ---------------------------------


def test_camera_summary_includes_transform_and_focal():
    cam = _fake_node(
        path="/obj/cam1", name="cam1", type_name="cam",
        parms={"focal": 50, "resx": 1920, "resy": 1080},
        parm_tuples={"t": (0, 1.5, 6), "r": (-10, 0, 0)},
    )
    hou = _fake_hou(obj_children=[cam])
    out = inspect_scene(hou)

    assert len(out["cameras"]) == 1
    c = out["cameras"][0]
    assert c["path"] == "/obj/cam1"
    assert c["focal"] == 50.0
    assert c["transform"] == [0.0, 1.5, 6.0]
    assert c["rotate"] == [-10.0, 0.0, 0.0]
    assert c["resolution"] == [1920, 1080]


def test_light_summary_handles_versioned_type_names():
    key = _fake_node(
        path="/obj/key", name="key", type_name="hlight::2.0",
        parms={"light_intensity": 2.5, "light_type": 7},
        parm_tuples={"t": (0, 5, 0), "r": (-30, 30, 0)},
    )
    env = _fake_node(path="/obj/env", name="env", type_name="envlight::2.0")
    hou = _fake_hou(obj_children=[key, env])

    out = inspect_scene(hou)
    paths = [l["path"] for l in out["lights"]]
    assert paths == ["/obj/key", "/obj/env"]
    k = out["lights"][0]
    assert k["intensity"] == 2.5
    assert k["light_type"] == 7
    assert k["transform"] == [0.0, 5.0, 0.0]


def test_geo_container_summary_includes_prim_count():
    sop = _fake_node(
        path="/obj/geo1/box1", name="box1", type_name="box",
        geometry_prim_count=12,
    )
    geo = _fake_node(
        path="/obj/geo1", name="geo1", type_name="geo",
        parm_tuples={"t": (1, 0, 0)},
        display_node=sop,
    )
    hou = _fake_hou(obj_children=[geo])

    out = inspect_scene(hou)
    assert len(out["geo_containers"]) == 1
    g = out["geo_containers"][0]
    assert g["path"] == "/obj/geo1"
    assert g["display_node"] == "/obj/geo1/box1"
    assert g["primitive_count"] == 12


def test_filters_are_disjoint():
    """A camera should not appear under lights or geo_containers, etc."""
    geo = _fake_node(path="/obj/geo1", name="geo1", type_name="geo")
    cam = _fake_node(path="/obj/cam1", name="cam1", type_name="cam")
    light = _fake_node(path="/obj/key", name="key", type_name="hlight::2.0")
    hou = _fake_hou(obj_children=[geo, cam, light])

    out = inspect_scene(hou)
    assert {c["path"] for c in out["cameras"]} == {"/obj/cam1"}
    assert {l["path"] for l in out["lights"]} == {"/obj/key"}
    assert {g["path"] for g in out["geo_containers"]} == {"/obj/geo1"}


def test_inspect_survives_partial_node_failure():
    good = _fake_node(path="/obj/geo1", name="geo1", type_name="geo")
    bad = SimpleNamespace(
        path=lambda: (_ for _ in ()).throw(RuntimeError("deleted mid-iteration")),
        name=lambda: "bad",
        type=lambda: SimpleNamespace(name=lambda: "geo", category=lambda: None),
        children=lambda: (),
        parmTuple=lambda _n: None,
        parm=lambda _n: None,
        displayNode=lambda: None,
    )
    hou = _fake_hou(obj_children=[good, bad])

    out = inspect_scene(hou)
    paths = [o["path"] for o in out["objects"]]
    assert paths == ["/obj/geo1"]  # bad one dropped, no crash


def test_inspect_handles_missing_obj_root():
    hou = SimpleNamespace(
        frame=lambda: 1.0,
        fps=lambda: 24.0,
        hipFile=SimpleNamespace(path=lambda: ""),
        node=lambda _p: None,
        selectedNodes=lambda: (),
    )
    out = inspect_scene(hou)
    assert out["available"] is True
    assert out["objects"] == []
    assert out["cameras"] == []
    assert out["lights"] == []
    assert out["geo_containers"] == []
