"""Collect a snapshot of the Houdini scene for the LLM.

Runs INSIDE Houdini (in the receiver process). The host (Ai-Bridge_Houdini
CLI) calls this via ``{"type": "inspect_scene"}`` to get the current scene
state, then forwards it to the LLM as additional context.

The module deliberately accepts the ``hou`` module as an argument rather
than importing it at module scope so:
  * tests can pass a ``SimpleNamespace`` stand-in,
  * importing this file outside Houdini does not crash, and
  * the receiver remains free to deploy this file alone alongside
    houdini_receiver.py (no third-party dependencies).

Every accessor is wrapped in defensive try/except: a partially-loaded
scene, a node missing a parm, or a deleted node mid-iteration must never
crash inspection. Failures are logged and the affected entry is dropped.
"""

from __future__ import annotations

import logging
from typing import Any


log = logging.getLogger("aibridge.houdini.inspection")


# Type names from /obj. Houdini exposes versioned names like "hlight::2.0";
# we match either by exact name or by the un-versioned prefix.
_CAMERA_TYPES: frozenset[str] = frozenset({"cam", "stereocam"})
_LIGHT_TYPE_PREFIXES: tuple[str, ...] = (
    "hlight",
    "envlight",
    "vraylight",
    "asadlight",
    "geolight",
    "indirectlight",
    "spotlight",
    "arnold_light",
    "redshift_light",
)
_GEO_TYPES: frozenset[str] = frozenset({"geo"})


def inspect_scene(hou: Any | None) -> dict:
    """Return a JSON-serializable snapshot of the current Houdini scene."""
    if hou is None:
        return _empty(error="hou module not available")

    try:
        return {
            "available": True,
            "error": None,
            "frame": _frame(hou),
            "fps": _fps(hou),
            "hip_file": _hip_file(hou),
            "objects": _list_obj_children(hou),
            "selected_nodes": _list_selected(hou),
            "cameras": _filter_obj_children(hou, _is_camera, _camera_summary),
            "lights": _filter_obj_children(hou, _is_light, _light_summary),
            "geo_containers": _filter_obj_children(hou, _is_geo, _geo_summary),
        }
    except Exception as e:
        log.exception("inspect_scene failed at top level")
        return _empty(error=f"inspection failed: {e}")


# ---- top-level fields ---------------------------------------------------


def _empty(*, error: str | None) -> dict:
    return {
        "available": False,
        "error": error,
        "frame": None,
        "fps": None,
        "hip_file": None,
        "objects": [],
        "selected_nodes": [],
        "cameras": [],
        "lights": [],
        "geo_containers": [],
    }


def _frame(hou: Any) -> float | None:
    try:
        return float(hou.frame())
    except Exception as e:
        log.warning("could not read frame: %s", e)
        return None


def _fps(hou: Any) -> float | None:
    try:
        return float(hou.fps())
    except Exception:
        return None


def _hip_file(hou: Any) -> str | None:
    try:
        return hou.hipFile.path()
    except Exception:
        return None


def _list_obj_children(hou: Any) -> list[dict]:
    obj = _safe_node(hou, "/obj")
    if obj is None:
        return []
    out: list[dict] = []
    for child in _safe_children(obj):
        summary = _node_summary(child)
        if summary is not None:
            out.append(summary)
    return out


def _list_selected(hou: Any) -> list[dict]:
    try:
        nodes = list(hou.selectedNodes())
    except Exception as e:
        log.warning("could not read selection: %s", e)
        return []
    return [s for s in (_node_summary(n) for n in nodes) if s is not None]


def _filter_obj_children(hou: Any, predicate, build) -> list[dict]:
    obj = _safe_node(hou, "/obj")
    if obj is None:
        return []
    out: list[dict] = []
    for child in _safe_children(obj):
        try:
            if not predicate(child):
                continue
        except Exception:
            continue
        summary = build(child)
        if summary is not None:
            out.append(summary)
    return out


# ---- predicates ---------------------------------------------------------


def _type_name(node: Any) -> str:
    try:
        return node.type().name() or ""
    except Exception:
        return ""


def _is_camera(node: Any) -> bool:
    name = _type_name(node).split("::", 1)[0]
    return name in _CAMERA_TYPES


def _is_light(node: Any) -> bool:
    name = _type_name(node).split("::", 1)[0]
    return any(name == p or name.startswith(p) for p in _LIGHT_TYPE_PREFIXES)


def _is_geo(node: Any) -> bool:
    name = _type_name(node).split("::", 1)[0]
    return name in _GEO_TYPES


# ---- summaries ----------------------------------------------------------


def _node_summary(node: Any) -> dict | None:
    try:
        type_name = _type_name(node)
        category = ""
        try:
            cat = node.type().category()
            category = cat.name() if cat is not None else ""
        except Exception:
            pass
        return {
            "path": node.path(),
            "name": node.name(),
            "type": type_name,
            "category": category,
        }
    except Exception as e:
        log.debug("could not summarize node: %s", e)
        return None


def _camera_summary(node: Any) -> dict | None:
    base = _node_summary(node)
    if base is None:
        return None
    base["transform"] = _parm_tuple(node, "t")
    base["rotate"] = _parm_tuple(node, "r")
    base["focal"] = _parm_float(node, "focal")
    base["resolution"] = _parm_resolution(node)
    return base


def _light_summary(node: Any) -> dict | None:
    base = _node_summary(node)
    if base is None:
        return None
    base["transform"] = _parm_tuple(node, "t")
    base["rotate"] = _parm_tuple(node, "r")
    base["intensity"] = _parm_float(node, "light_intensity")
    base["light_type"] = _parm_int(node, "light_type")
    return base


def _geo_summary(node: Any) -> dict | None:
    base = _node_summary(node)
    if base is None:
        return None
    base["transform"] = _parm_tuple(node, "t")
    base["display_node"] = _display_node_path(node)
    base["primitive_count"] = _safe_prim_count(node)
    return base


# ---- low-level safe accessors ------------------------------------------


def _safe_node(hou: Any, path: str) -> Any | None:
    try:
        return hou.node(path)
    except Exception:
        return None


def _safe_children(node: Any) -> list[Any]:
    try:
        return list(node.children())
    except Exception:
        return []


def _parm_tuple(node: Any, name: str) -> list[float] | None:
    try:
        pt = node.parmTuple(name)
        if pt is None:
            return None
        return [float(v) for v in pt.eval()]
    except Exception:
        return None


def _parm_float(node: Any, name: str) -> float | None:
    try:
        p = node.parm(name)
        return float(p.eval()) if p is not None else None
    except Exception:
        return None


def _parm_int(node: Any, name: str) -> int | None:
    try:
        p = node.parm(name)
        return int(p.eval()) if p is not None else None
    except Exception:
        return None


def _parm_resolution(node: Any) -> list[int] | None:
    try:
        x = node.parm("resx")
        y = node.parm("resy")
        if x is None or y is None:
            return None
        return [int(x.eval()), int(y.eval())]
    except Exception:
        return None


def _display_node_path(geo_node: Any) -> str | None:
    try:
        d = geo_node.displayNode()
        return d.path() if d is not None else None
    except Exception:
        return None


def _safe_prim_count(geo_node: Any) -> int | None:
    """Best-effort: read the cached geometry on the display node."""
    try:
        d = geo_node.displayNode()
        if d is None:
            return None
        geo = d.geometry()
        return int(geo.intrinsicValue("primitivecount")) if geo is not None else None
    except Exception:
        return None
