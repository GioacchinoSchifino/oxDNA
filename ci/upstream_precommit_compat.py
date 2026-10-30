#!/usr/bin/env python3
"""Cross-platform adaptation of oxDNA's pre-commit checks.

The original hook checks staged filenames, an obsolete src/defs.h RELEASE
macro, and git whitespace rules. Modern oxDNA defines RELEASE in CMake.
This adapter checks the actual staged index and never publishes anything.
"""
from __future__ import annotations
import re
import subprocess
import sys
from pathlib import Path

def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=check)

def staged_file(path: str) -> str | None:
    # Do not fall back to HEAD: a staged deletion must fail closed.
    p = git("show", f":{path}", check=False)
    return p.stdout.decode("utf-8", errors="replace") if p.returncode == 0 else None

def main() -> int:
    try:
        root = git("rev-parse", "--show-toplevel").stdout.decode().strip()
        if Path.cwd().resolve() != Path(root).resolve():
            print(f"PREFLIGHT_FAIL: run at repository root: {root}", file=sys.stderr)
            return 2
        flag = git("config", "--bool", "hooks.allownonascii", check=False)
        allow = flag.returncode == 0 and flag.stdout.strip() == b"true"
        changes = git("diff", "--cached", "--name-only", "--diff-filter=A", "-z").stdout
        names = [b for b in changes.split(b"\x00") if b]
        bad = [name for name in names if not all(0x20 <= x < 0x7f for x in name)]
        if bad and not allow:
            for name in bad:
                print(f"PREFLIGHT_FAIL: non-ASCII/control staged filename {name!r}", file=sys.stderr)
            return 1
        whitespace = git("diff", "--cached", "--check", check=False)
        if whitespace.returncode != 0:
            print("PREFLIGHT_FAIL: git diff --cached --check", file=sys.stderr)
            print(whitespace.stdout.decode(errors="replace"), file=sys.stderr)
            print(whitespace.stderr.decode(errors="replace"), file=sys.stderr)
            return 1
        defs, cmake = staged_file("src/defs.h"), staged_file("CMakeLists.txt")
        if defs is None or cmake is None:
            print("PREFLIGHT_FAIL: missing staged src/defs.h or CMakeLists.txt", file=sys.stderr)
            return 1
        old = re.search(r'^\s*#\s*define\s+RELEASE\s+"([^"]+)"', defs, flags=re.MULTILINE)
        if old:
            tag = git("describe", "--tags", check=False)
            cur = tag.stdout.decode().strip()
            if tag.returncode != 0 or old.group(1) != cur:
                print(f"PREFLIGHT_FAIL: legacy RELEASE={old.group(1)!r} != git describe={cur!r}", file=sys.stderr)
                return 1
            policy = "legacy_defs_tag_match"
        else:
            # The compiled macro is -D'RELEASE="${tag}"', not in src/defs.h.
            if not re.search(r"ADD_DEFINITIONS\s*\([^\n]*RELEASE\\?=", cmake, flags=re.IGNORECASE):
                print("PREFLIGHT_FAIL: no CMake RELEASE build definition", file=sys.stderr)
                return 1
            if not re.search(r'set\s*\(\s*tag\s+', cmake, flags=re.IGNORECASE):
                print("PREFLIGHT_FAIL: no CMake tag assignment", file=sys.stderr)
                return 1
            m = re.search(r'set\s*\(\s*OXDNA_ANM_RELEASE_LABEL\s+"([^"]+)"', cmake, flags=re.IGNORECASE)
            if m and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+\-]*", m.group(1)):
                print("PREFLIGHT_FAIL: invalid derivative release label", file=sys.stderr)
                return 1
            policy = "cmake_build_time_release"
        print(f"OXDNA_UPSTREAM_PRECOMMIT_COMPAT=PASS policy={policy} staged_adds={len(names)}")
        return 0
    except (OSError, subprocess.CalledProcessError) as e:
        print(f"PREFLIGHT_ERROR: {e}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
