"""YAML data-layer loader.

Leaf reference data (hardware chips, models, datasets, …) lives as YAML and is
loaded + validated against a pydantic schema at import, then assembled into a
``Registry`` subclass whose consumer API is identical to the former
hand-written Python registry (``Hardware.Cloud.H100.memory.bandwidth`` etc.).

Two encodings are handled before pydantic validation:

* **Unit-bearing strings** (``"3.35 TB/s"``, ``"80 GiB"``) are parsed by pint —
  this is already handled by the ``Quantity`` validator in ``core/types.py``,
  so the loader passes strings straight through.
* **``@tech:`` references** (``"@tech:Interconnect.NVLink.latency"``) preserve
  the instance→tech-class single source of truth: the value lives once in
  ``Hardware.Tech`` and instances point at it rather than copying it. The loader
  resolves these against ``tech_root`` into the live Quantity object.

See the project MLSysIM rules → *Storage format*.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any
import yaml

from .registry import Registry

_TECH_PREFIX = "@tech:"
_PROV_PREFIX = "@prov:"


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(loader: _UniqueKeyLoader, node, deep: bool = False):
    """Mapping constructor that raises ``ValueError`` on duplicate keys.

    PyYAML's default behavior is last-key-wins, which would let a duplicated
    registry field silently shadow an earlier value; this makes that a load
    error instead, naming the offending file.
    """
    loader.flatten_mapping(node)
    mapping = {}
    source = getattr(getattr(loader, "stream", None), "name", "<yaml>")
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"duplicate YAML key {key!r} in {source}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _caller_module(default: str = __name__) -> str:
    """Module name of the caller's caller (two frames up).

    Used to stamp ``__module__`` on dynamically built Registry classes so
    they introspect (repr, pickle, docs) as belonging to the registry module
    that called ``load_registry``/``load_collection``, not to this loader.
    """
    frame = inspect.currentframe()
    caller = frame.f_back.f_back if frame and frame.f_back and frame.f_back.f_back else None
    return caller.f_globals.get("__name__", default) if caller else default


def _load_mapping(yaml_file) -> dict[str, Any]:
    """Load one YAML file as a dict using the duplicate-key-rejecting loader.

    Raises ``ValueError`` if the document is not a mapping (a bare list or
    scalar at top level would otherwise fail much later, inside pydantic).
    """
    path = Path(yaml_file)
    with path.open(encoding="utf-8") as fh:
        raw = yaml.load(fh, Loader=_UniqueKeyLoader)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return raw


def _resolve(node: Any, tech_root: Any) -> Any:
    """Recursively resolve reference markers.

    * ``@tech:PATH`` → dotted-attribute lookup against ``tech_root`` (preserves the
      instance→tech-class single source of truth).
    * ``@prov:KEY`` → the named ``Provenance`` in ``core/provenance_catalog`` (so
      a provenance record shared by many entries lives once in the catalog).
    """
    if isinstance(node, dict):
        return {k: _resolve(v, tech_root) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve(v, tech_root) for v in node]
    if isinstance(node, str):
        if node.startswith(_TECH_PREFIX):
            if tech_root is None:
                raise ValueError(f"{node!r} found but no tech_root supplied to load_registry")
            obj = tech_root
            for part in node[len(_TECH_PREFIX):].split("."):
                obj = getattr(obj, part)
            return obj
        if node.startswith(_PROV_PREFIX):
            from . import provenance_catalog as pc
            return getattr(pc, node[len(_PROV_PREFIX):])
    return node


def _instantiate(raw: dict, *, model_cls, type_map, tech_root):
    """Validate one entry dict into its pydantic model.

    Polymorphic schemas (e.g. the ``Workload`` family) carry a ``__type__``
    tag selecting the concrete class from ``type_map``; uniform schemas omit it
    and use ``model_cls``.
    """
    raw = dict(raw)
    type_name = raw.pop("__type__", None)
    if type_name is not None:
        if not type_map or type_name not in type_map:
            raise ValueError(f"unknown __type__ {type_name!r} (type_map keys: {sorted(type_map or [])})")
        cls = type_map[type_name]
    else:
        cls = model_cls
    if cls is None:
        raise ValueError("no model class: supply model_cls or a __type__ in the data")
    return cls(**_resolve(raw, tech_root))


def _build(entries: dict, *, name: str, doc: str, model_cls, type_map, tech_root, module: str) -> type:
    attrs: dict[str, Any] = {"__doc__": doc, "__module__": module}
    for key, raw in entries.items():
        attrs[key] = _instantiate(raw, model_cls=model_cls, type_map=type_map, tech_root=tech_root)
    return type(name, (Registry,), attrs)


def load_registry(data_dir, model_cls=None, *, name: str, doc: str = "",
                  tech_root: Any = None, type_map: dict | None = None,
                  module: str | None = None) -> type:
    """Build a ``Registry`` subclass from one-entry-per-file YAML in ``data_dir``.

    Each ``*.yaml`` becomes one validated instance, exposed as a class attribute
    named by the file's ``__key__`` field (falling back to the file stem). Files
    are read sorted; ``Registry.list()`` and attribute access are
    order-independent, so this matches the former definition order. Used for
    per-chip hardware data.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"registry data directory not found: {data_dir}")
    entries: dict[str, Any] = {}
    seen_files: dict[str, Path] = {}
    yaml_files = sorted(data_dir.glob("*.yaml"))
    if not yaml_files:
        raise ValueError(f"registry data directory has no YAML files: {data_dir}")
    for yaml_file in yaml_files:
        raw = _load_mapping(yaml_file)
        key = raw.pop("__key__", yaml_file.stem)
        if key in entries:
            raise ValueError(f"duplicate registry key {key!r} in {yaml_file} and {seen_files[key]}")
        entries[key] = raw
        seen_files[key] = yaml_file
    return _build(
        entries, name=name, doc=doc, model_cls=model_cls, type_map=type_map,
        tech_root=tech_root, module=module or _caller_module(),
    )


def load_sourced_registry(yaml_file, *, name: str, doc: str = "", module: str | None = None) -> type:
    """Build a ``Registry`` of provenance-carrying scalars from a YAML file.

    For literature/anchor registries whose entries are ``sourced(value, prov, …)``
    scalars (a ``float`` subclass carrying provenance) rather than pydantic models.
    Each entry must be a mapping
    ``{value, provenance: <catalog-key>, name?, description?}`` and is rebuilt via
    ``sourced(value, provenance_catalog.<key>, …)``. Bare scalars are rejected so
    sourced registries cannot silently bypass provenance.
    """
    from .provenance import sourced
    from . import provenance_catalog as pc

    raw = _load_mapping(yaml_file)
    attrs: dict[str, Any] = {"__doc__": doc, "__module__": module or _caller_module()}
    for key, v in raw.items():
        if not isinstance(v, dict) or "provenance" not in v or "value" not in v:
            raise ValueError(
                f"{yaml_file}:{key} must define value and provenance for sourced registries"
            )
        attrs[key] = sourced(
            v["value"], getattr(pc, v["provenance"]),
            name=v.get("name", ""), description=v.get("description", ""),
        )
    return type(name, (Registry,), attrs)


def load_collection(yaml_file, model_cls=None, *, name: str, doc: str = "",
                    tech_root: Any = None, type_map: dict | None = None,
                    module: str | None = None) -> type:
    """Build a ``Registry`` subclass from a single per-category YAML file.

    The file is a mapping ``{attr_name: entry_dict}``; each entry validates into
    ``model_cls`` (or its ``__type__`` from ``type_map`` for polymorphic
    schemas). Used for per-category model data, where the cohesive set is more
    useful read together than split across files.
    """
    raw = _load_mapping(yaml_file)
    return _build(
        raw, name=name, doc=doc, model_cls=model_cls, type_map=type_map,
        tech_root=tech_root, module=module or _caller_module(),
    )
