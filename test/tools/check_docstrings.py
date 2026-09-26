#!/usr/bin/env python3
"""Gate: every exported ChemMechSim symbol has a docstring above its definition.

Usage: python3 test/tools/check_docstrings.py
Exit 0 + "OK: N/N exported symbols have docstrings" when complete; exit 1 + MISSING list.
Matches definitions of the forms: function/struct/mutable struct/abstract type/const NAME,
and short-form `NAME(...) = ...` / `NAME = ...` at any indentation.
A docstring is a triple-quote or single-quote line (after optional whitespace)
above the definition,
allowing blank lines, comments and @-annotations in between. Heuristic by design; the
strict Documenter build (PR-2) is the final enforcement.
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


def exported_names():
    out = subprocess.run(
        ["julia", "--project=.", "-e",
         "using ChemMechSim; println.(sort(string.(names(ChemMechSim))))"],
        capture_output=True, text=True, cwd=ROOT, check=True)
    return [n for n in out.stdout.split() if n != "ChemMechSim"]


DEF = re.compile(
    r'^\s*(?:function|struct|mutable struct|abstract type|const)\s+([A-Za-z_]\w*)'
    r'|^\s*([A-Za-z_]\w*)\s*\([^{}\n]*\)\s*='      # short-form def: name(args) = body
    r'|^\s*([A-Za-z_]\w*)\s*=[^=]')                    # plain assignment: name = value


def scan_file(path):
    """name -> index of its first definition line."""
    lines = path.read_text().splitlines()
    defs = {}
    for i, line in enumerate(lines):
        m = DEF.match(line)
        if not m:
            continue
        name = m.group(1) or m.group(2) or m.group(3)
        if name and not name.startswith("_"):
            defs.setdefault(name, i)
    return lines, defs


def has_docstring(lines, i):
    j = i - 1
    while j >= 0:
        s = lines[j].strip()
        if s == "":
            j -= 1
            continue
        if s.startswith("#") or s.startswith("@"):
            j -= 1
            continue
        # docstring opener (`"""`), quoted continuation (`"..." *`), or a closing
        # line of a multi-line block (text ending in `"""` or `”`).
        if s.startswith('"""') or s.startswith('"') or s.endswith('"""') or s.endswith('"'):
            return True
        return False
    return False


def main():
    per_file = {f: scan_file(f) for f in sorted(SRC.rglob("*.jl"))}
    exports = exported_names()
    missing = [n for n in exports
               if not any(n in defs and has_docstring(lines, defs[n])
                          for lines, defs in per_file.values())]
    if missing:
        print(f"MISSING ({len(missing)}): {', '.join(sorted(missing))}")
        sys.exit(1)
    print(f"OK: {len(exports)}/{len(exports)} exported symbols have docstrings")


if __name__ == "__main__":
    main()
