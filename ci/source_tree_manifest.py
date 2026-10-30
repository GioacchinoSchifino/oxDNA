#!/usr/bin/env python3
"""Fingerprint all regular source files, rejecting symbolic links.

Print stable JSON to stdout; keep output outside the source directory.
Only .git metadata is excluded. Run before compilation/testing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path)
args = parser.parse_args()
root = args.source.resolve(strict=True)
records = []
for path in sorted(root.rglob("*")):
    relative = path.relative_to(root)
    if ".git" in relative.parts:
        continue
    if path.is_symlink():
        sys.exit("STOP_SOURCE_SYMLINK: " + str(relative))
    if path.is_file():
        records.append({"path": relative.as_posix(),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    elif not path.is_dir():
        sys.exit("STOP_SOURCE_SPECIAL_FILE: " + str(relative))
canonical = "".join(x["sha256"] + "  " + x["path"] + "\n" for x in records)
print(json.dumps({"format": "sha256-two-spaces-path-newline-v1",
                  "tree_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
                  "files": records}, indent=2, allow_nan=False))
