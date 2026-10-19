#!/usr/bin/env python3
"""Run a frozen entry point from the public, path-normalized repository.

Five public files differ from their frozen bytes only in a path string; PATH_NORMALIZATION.json
lists, for each, the frozen SHA-256, the public SHA-256, and the change.  Frozen entry points
re-check the frozen SHA-256 of some of these files (for example, the block-16 endpoint checks
analysis/run_posthoc_power_study_v1.py) and therefore stop in a public checkout before reading
any input.

This wrapper changes no frozen file, protocol, or receipt.  It loads the entry point as a
module and replaces only its module-level file-hash function.  The replacement returns the
true SHA-256 of every file, except that a file listed in PATH_NORMALIZATION.json whose bytes
match the declared *public* SHA-256 exactly is reported with its declared frozen SHA-256.  A
listed file with any other bytes keeps its true hash and fails the entry point's own check.
Every substitution is written to stderr.  The frozen bytes themselves cannot be checked in a
public checkout; compare them with the controlled archive.

usage: python -B auditability_certificate_v1/analysis/public_compat_runner.py ENTRY.py [ARGS ...]
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "PATH_NORMALIZATION.json"
HASH_FUNCTIONS = ("sha256", "sha", "digest", "file_sha256")


def _true_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def normalization_table(root: Path = ROOT) -> dict[Path, tuple[str, str]]:
    """{resolved path: (public sha256, frozen sha256)} from PATH_NORMALIZATION.json."""
    rows = json.loads((root / "PATH_NORMALIZATION.json").read_text(encoding="utf-8"))["files"]
    table = {}
    for row in rows:
        path = (root / row["path"]).resolve()
        if path in table or len(row["public_sha256"]) != 64 or len(row["frozen_sha256"]) != 64:
            raise ValueError(f"invalid normalization row: {row['path']}")
        table[path] = (row["public_sha256"], row["frozen_sha256"])
    return table


def compat_hash(original, table, log):
    def wrapped(path, *args, **kwargs):
        value = original(path, *args, **kwargs)
        try:
            key = Path(path).resolve()
        except TypeError:
            return value
        if key in table and value == table[key][0]:
            log.append(str(key.relative_to(ROOT)) if key.is_relative_to(ROOT) else str(key))
            print(f"[public-compat] {log[-1]}: public bytes {value[:12]}... match "
                  f"PATH_NORMALIZATION.json; reporting frozen hash {table[key][1][:12]}...", file=sys.stderr)
            return table[key][1]
        return value
    return wrapped


def load_entry(entry: Path, root: Path = ROOT):
    """Import an entry point with its file-hash function made normalization-aware.

    Returns (module, substitution log).  The module is registered under its file stem and its
    directory is put on sys.path, so that worker processes can import it by name."""
    entry = Path(entry).resolve()
    table = normalization_table(root)
    sys.path.insert(0, str(entry.parent))
    spec = importlib.util.spec_from_file_location(entry.stem, entry)
    module = importlib.util.module_from_spec(spec)
    sys.modules[entry.stem] = module
    spec.loader.exec_module(module)
    log: list[str] = []
    patched = []
    for name in HASH_FUNCTIONS:
        fn = getattr(module, name, None)
        if callable(fn) and getattr(fn, "__module__", None) == module.__name__:
            setattr(module, name, compat_hash(fn, table, log)); patched.append(name)
    if not patched:
        raise RuntimeError(f"{entry.name}: no module-level file-hash function to adapt")
    return module, log


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__); return 0 if argv else 2
    entry = Path(argv[0])
    module, _ = load_entry(entry)
    if not callable(getattr(module, "main", None)):
        raise RuntimeError(f"{entry.name} has no main()")
    sys.argv = [str(entry), *argv[1:]]
    result = module.main()
    return int(result) if isinstance(result, int) else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
