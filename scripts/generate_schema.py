import os
import yaml
import numpy as np
from ase2sprkkr import InputParameters
import ase2sprkkr.common.grammar_types as gt

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "sprkkr2nomad", "schemas")
os.makedirs(OUT_DIR, exist_ok=True)

# Sets of class names (taken directly from provided sources)
INT_TYPES      = {"Integer", "Unsigned", "FixedPointNumber", "ObjectNumber"}
FLOAT_TYPES    = {"Real", "Energy", "RealWithUnits", "BaseRealWithUnits", "Number"}
BOOL_TYPES     = {"Bool", "Boolean", "IntBool", "Flag", "BaseBool"}
STRING_TYPES   = {"String", "QString", "AlwaysQString", "LineString", "BaseString"}
ENUM_TYPES     = {"Keyword"}               # DefKeyword produces Keyword underneath
COMPLEX_TYPES  = {"Complex"}
ARRAY_TYPES    = {"Array", "SetOf"}
SEQUENCE_TYPES = {"Sequence"}
TABLE_TYPES    = {"Table"}
RANGE_TYPES    = {"Range"}
MIXED_TYPES    = {"Mixed", "PotMixed", "Variant", "CustomMixed"}

def norm(name: str) -> str:
    return "".join(p.capitalize() for p in name.replace("_", " ").split())

def is_section(defn) -> bool:
    return getattr(defn, "item_type", None) == "section"

def is_value(defn) -> bool:
    return getattr(defn, "item_type", None) == "value"

def kind(g) -> str:
    cn = g.__class__.__name__
    if cn in INT_TYPES: return "int"
    if cn in FLOAT_TYPES: return "float"
    if cn in BOOL_TYPES: return "bool"
    if cn in STRING_TYPES: return "string"
    if cn in ENUM_TYPES: return "string"
    if cn in COMPLEX_TYPES: return "string"  # portable fallback
    if cn in RANGE_TYPES:
        inner = getattr(g, "_type", None)
        return kind(inner) if inner else "string"
    if cn in MIXED_TYPES: return "string"
    if cn in ARRAY_TYPES:
        inner = getattr(g, "type", None)
        return kind(inner) if inner else "string"
    if cn in SEQUENCE_TYPES:
        types = getattr(g, "types", [])
        if types and all(kind(t) == kind(types[0]) for t in types):
            return kind(types[0])
        return "string"
    if cn in TABLE_TYPES:
        seq = getattr(g, "sequence", None)
        if seq and getattr(seq, "types", []):
            ks = {kind(t) for t in seq.types}
            return ks.pop() if len(ks) == 1 else "string"
        return "string"
    return "string"

def enum_values(g):
    cn = g.__class__.__name__
    if cn in ENUM_TYPES:
        # Prefer keywords; fall back to choices keys if present
        vals = list(getattr(g, "keywords", []))
        if not vals and hasattr(g, "choices"):
            vals = list(getattr(g, "choices").keys())
        return vals or None
    return None

def number_constraints(g):
    out = {}
    if hasattr(g, "min") and g.min is not None:
        out["min_value"] = g.min
    if hasattr(g, "max") and g.max is not None:
        out["max_value"] = g.max
    return out

def range_annotation(g):
    if g.__class__.__name__ in RANGE_TYPES:
        return {"is_range": True}
    return {}

def units_info(g):
    if hasattr(g, "units"):
        units = getattr(g, "units")
        if isinstance(units, dict) and units:
            default_unit = getattr(g, "default_unit", None)
            return {
                "unit": default_unit or list(units.keys())[0],
                "allowed_units": list(units.keys())
            }
    return {}

def shape_info(g):
    cn = g.__class__.__name__
    if cn in ARRAY_TYPES:
        min_len = getattr(g, "min_length", None)
        max_len = getattr(g, "max_length", None)
        if isinstance(min_len, int) and isinstance(max_len, int) and min_len == max_len:
            return {"shape": [min_len]}
        return {}
    if cn in SEQUENCE_TYPES:
        ln = len(getattr(g, "types", []))
        return {"shape": [ln]}
    if cn in TABLE_TYPES:
        seq = getattr(g, "sequence", None)
        if seq:
            cols = len(getattr(seq, "types", []))
            return {"table_columns": cols}
    if hasattr(g, "shape") and isinstance(getattr(g, "shape"), (tuple, list)):
        shp = list(getattr(g, "shape"))
        return {"shape": shp}
    return {}

def repeated_info(vdef, g):
    rep = getattr(vdef, "is_repeated", None)
    if rep:
        r = {}
        if getattr(rep, "is_array", False):
            r["repeated"] = True
        if getattr(rep, "is_dict", False):
            r["repeated_dict"] = True
        return r
    if g.__class__.__name__ in ARRAY_TYPES | SEQUENCE_TYPES | TABLE_TYPES:
        return {"repeated": True}
    return {}

def missing_value_info(g):
    if hasattr(g, "missing_value"):
        try:
            can_omit, default_val, _supp = g.missing_value()
            out = {}
            if can_omit:
                out["optional"] = True
            if default_val not in (None, False):
                out["default_if_missing"] = to_serializable(default_val)
            return out
        except Exception:
            return {}
    return {}

def default_value(vdef, g):
    dv = getattr(vdef, "default_value", None)
    if dv is None:
        dv = getattr(g, "default_value", None)
    if callable(dv):
        return None
    return dv

def description(vdef, g):
    d = getattr(vdef, "_description", None) or getattr(g, "_description", None)
    if not d and hasattr(g, "additional_description"):
        try:
            ad = g.additional_description()
            if ad:
                d = ad
        except Exception:
            pass
    return d.strip() if isinstance(d, str) else None

def to_serializable(v):
    try:
        import unyt
        unyt_types = (unyt.unyt_quantity, unyt.unyt_array)
    except Exception:
        unyt_types = ()
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [to_serializable(x) for x in v]
    if isinstance(v, set):
        return [to_serializable(x) for x in sorted(v, key=lambda x: str(x))]
    if unyt_types and isinstance(v, unyt_types):
        return float(v.value)
    # Dicts (e.g. choices) should not appear as default; stringify if they do
    if isinstance(v, dict):
        return {str(k): to_serializable(vv) for k, vv in v.items()}
    return str(v)

def build_section(defn, root_prefix):
    sections = {}
    section_name = f"{root_prefix}{norm(defn.name)}"
    sec = {
        "base_section": "nomad.datamodel.metainfo.basesections.EntryData",
        "description": f"Auto-generated SPRKKR section '{defn.name}'",
        "quantities": {},
        "sub_sections": {}
    }
    members = getattr(defn, "_members", {}) or {}
    for child_name, child_def in members.items():
        if is_section(child_def):
            child_sections = build_section(child_def, section_name)
            sections.update(child_sections)
            child_top = f"{section_name}{norm(child_def.name)}"
            sec["sub_sections"][child_def.name] = {
                "section": child_top,
                "repeats": bool(getattr(child_def, "is_repeated", False))
            }
            continue
        if is_value(child_def):
            g = getattr(child_def, "type", None)
            if g is None:
                continue
            q = {"type": {"type_kind": kind(g)}}
            ev = enum_values(g)
            if ev:
                q["enum"] = [to_serializable(x) for x in ev]
            q.update(number_constraints(g))
            q.update(range_annotation(g))
            q.update(units_info(g))
            q.update(shape_info(g))
            q.update(repeated_info(child_def, g))
            q.update(missing_value_info(g))
            dv = default_value(child_def, g)
            if dv is not None:
                q["default"] = to_serializable(dv)
            dsc = description(child_def, g)
            if dsc:
                q["description"] = dsc
            if getattr(child_def, "is_fixed", False):
                q["read_only"] = True
            # Clean empty shape
            if q.get("shape") is None:
                q.pop("shape", None)
            sec["quantities"][child_name] = q
    if not sec["sub_sections"]:
        sec.pop("sub_sections")
    sections[section_name] = sec
    return sections

def generate_one(root_name, root_def):
    pkg = f"sprkkr_{root_name.lower()}"
    sections = build_section(root_def, f"Sprkkr{norm(root_name)}")
    schema = {"schema_package": pkg, "sections": sections}
    out_path = os.path.join(OUT_DIR, f"{root_name.lower()}.schema.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(schema, f, sort_keys=False)
    print("Wrote", out_path)

def main():
    for name, defn in InputParameters.definitions.items():
        generate_one(name, defn)

if __name__ == "__main__":
    main()