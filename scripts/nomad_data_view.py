#!/usr/bin/env python3
"""Parse files with the local NOMAD SPRKKR parser and output a data-only view.

This intentionally removes NOMAD/metainfo bookkeeping (e.g. keys starting with
"m_") to provide a compact JSON tree that is easier to inspect.

Examples:
  uv run python scripts/nomad_data_view.py examples/Fe_Scf/Fe_SCF.out
  uv run python scripts/nomad_data_view.py examples/Fe_Scf/*.out --pretty
  uv run python scripts/nomad_data_view.py examples/Fe_Scf/*.out --out out/

Notes:
- By default, outputs only the parsed archive `data` section.
- Use --archive to output the full archive (still stripped of metainfo keys).
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from nomad.parsing import parsers as nomad_parsers


LOGGER = logging.getLogger("nomad_data_view")


META_KEY_PREFIXES = ("m_",)


def _strip_metainfo(value: Any, *, drop_refs: bool) -> Any:
    """Remove NOMAD/metainfo keys and optionally drop reference fields."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, child in value.items():
            if isinstance(key, str) and key.startswith(META_KEY_PREFIXES):
                continue
            if drop_refs and isinstance(key, str) and key.endswith("_ref"):
                continue
            out[key] = _strip_metainfo(child, drop_refs=drop_refs)
        return out
    if isinstance(value, list):
        return [_strip_metainfo(v, drop_refs=drop_refs) for v in value]
    return value


def _expand_paths(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        expanded = glob.glob(pattern)
        if expanded:
            paths.extend(expanded)
        else:
            # Keep literal path if it exists (e.g. no glob chars)
            if os.path.exists(pattern):
                paths.append(pattern)
            else:
                raise FileNotFoundError(f"No files matched: {pattern}")

    # De-dup while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for path in paths:
        real = os.path.realpath(path)
        if real not in seen:
            seen.add(real)
            unique.append(path)
    return unique


@dataclass(frozen=True)
class OutputTarget:
    to_stdout: bool
    path: Path | None


def _resolve_output_target(out: str | None, multiple_inputs: bool) -> OutputTarget:
    if out is None or out == "-":
        return OutputTarget(to_stdout=True, path=None)

    out_path = Path(out)
    if multiple_inputs and out_path.exists() and out_path.is_file():
        raise ValueError("--out points to a file, but multiple inputs were provided. Use a directory or '-' for stdout.")
    return OutputTarget(to_stdout=False, path=out_path)


def _write_json(target: OutputTarget, input_path: str, payload: Any, *, pretty: bool) -> None:
    dump_kwargs = {"ensure_ascii": False}
    if pretty:
        dump_kwargs.update({"indent": 2, "sort_keys": False})

    if target.to_stdout:
        print(json.dumps(payload, **dump_kwargs))
        return

    assert target.path is not None
    out_path = target.path
    if out_path.exists() and out_path.is_dir():
        base = Path(input_path).name
        out_file = out_path / f"{base}.data.json"
    elif str(out_path).endswith(os.sep) or (not out_path.suffix and not out_path.exists()):
        # Treat as directory path even if it doesn't exist yet.
        out_path.mkdir(parents=True, exist_ok=True)
        base = Path(input_path).name
        out_file = out_path / f"{base}.data.json"
    else:
        # Single explicit output file.
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_file = out_path

    out_file.write_text(json.dumps(payload, **dump_kwargs) + ("\n" if pretty else ""), encoding="utf-8")


def parse_one(mainfile: str, *, full_archive: bool, drop_refs: bool) -> Any:
    parser, mainfile_keys = nomad_parsers.match_parser(mainfile)
    if parser is None:
        raise ValueError(f"No NOMAD parser matched: {mainfile}")

    archives = nomad_parsers.run_parser(
        mainfile, parser, mainfile_keys, logger=LOGGER
    )
    archive = archives[0]

    raw = (
        archive.m_to_dict(with_meta=False, resolve_references=False)
        if full_archive
        else archive.data.m_to_dict(with_meta=False, resolve_references=False)
    )
    return _strip_metainfo(raw, drop_refs=drop_refs)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run SPRKKR NOMAD parser and print a stripped data-only JSON tree.")
    ap.add_argument("inputs", nargs="+", help="Input files or globs (e.g. examples/Fe_Scf/*.out)")
    ap.add_argument("--out", default=None, help="Output path: '-' (stdout), a file, or a directory")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    ap.add_argument("--archive", action="store_true", help="Output full archive instead of only archive.data")
    ap.add_argument(
        "--keep-refs",
        action="store_true",
        help="Keep NOMAD reference fields like 'model_system_ref' (default: drop all *_ref keys)",
    )
    ap.add_argument("--log", default="WARNING", help="Log level (DEBUG, INFO, WARNING, ERROR)")

    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, str(args.log).upper(), logging.WARNING))

    input_paths = _expand_paths(args.inputs)
    target = _resolve_output_target(args.out, multiple_inputs=len(input_paths) > 1)

    if target.to_stdout and len(input_paths) > 1:
        # Keep stdout machine-parseable: emit a dict keyed by input path.
        combined: dict[str, Any] = {}
        for path in input_paths:
            combined[path] = parse_one(
                path, full_archive=args.archive, drop_refs=(not args.keep_refs)
            )
        _write_json(target, input_paths[0], combined, pretty=args.pretty)
    else:
        for path in input_paths:
            payload = parse_one(
                path, full_archive=args.archive, drop_refs=(not args.keep_refs)
            )
            _write_json(target, path, payload, pretty=args.pretty)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
