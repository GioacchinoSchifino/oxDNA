#!/usr/bin/env python3
"""Test the real Git hook using disposable repositories and commits."""
from __future__ import annotations
from pathlib import Path
import subprocess
import sys
import tempfile

HOOK = Path(__file__).resolve().parent.parent / '.githooks/pre-commit'
CHECK = Path(__file__).resolve().parent / 'upstream_precommit_compat.py'

def run(root: Path, *cmd: str, ok: bool | None = True):
    p = subprocess.run(cmd, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if ok is not None and (p.returncode == 0) != ok:
        raise RuntimeError(f'{cmd}: exit={p.returncode}: {p.stdout[:1200]}')
    return p

count = 0
with tempfile.TemporaryDirectory(prefix='oxdna-precommit-test-') as d:
    root = Path(d)
    run(root, 'git', 'init', '-q')
    run(root, 'git', 'config', 'user.email', 'ci-test@example.invalid')
    run(root, 'git', 'config', 'user.name', 'CI Test')
    for directory in ('src', 'ci', '.githooks'):
        (root / directory).mkdir()
    (root / 'ci/upstream_precommit_compat.py').write_bytes(CHECK.read_bytes())
    (root / '.githooks/pre-commit').write_bytes(HOOK.read_bytes())
    (root / '.githooks/pre-commit').chmod(0o755)
    run(root, 'git', 'config', 'core.hooksPath', '.githooks')
    (root / 'src/defs.h').write_text('// modern oxDNA: release comes from CMake\n')
    valid = 'set(tag "v1.2.3")\nADD_DEFINITIONS(-D\'RELEASE="${tag}"\')\n'
    (root / 'CMakeLists.txt').write_text(valid)
    run(root, 'git', 'add', '.')
    p = run(root, 'git', 'commit', '-m', 'baseline')
    assert 'OXDNA_UPSTREAM_PRECOMMIT_COMPAT=PASS' in p.stdout
    count += 1
    (root / 'good.txt').write_text('good\n')
    run(root, 'git', 'add', 'good.txt')
    run(root, 'git', 'commit', '-m', 'ASCII filename')
    count += 1
    (root / 'badé.txt').write_text('ok\n')
    run(root, 'git', 'add', 'badé.txt')
    p = run(root, 'git', 'commit', '-m', 'nonascii rejected', ok=False)
    assert 'non-ASCII' in p.stdout
    count += 1
    run(root, 'git', 'config', 'hooks.allownonascii', 'true')
    run(root, 'git', 'commit', '-m', 'nonascii allowed')
    count += 1
    (root / 'whitespace.txt').write_text('bad trailing   \n')
    run(root, 'git', 'add', 'whitespace.txt')
    p = run(root, 'git', 'commit', '-m', 'whitespace rejected', ok=False)
    assert 'whitespace' in p.stdout or 'trailing' in p.stdout
    count += 1
    (root / 'whitespace.txt').write_text('no trailing\n')
    run(root, 'git', 'add', 'whitespace.txt')
    run(root, 'git', 'commit', '-m', 'whitespace fixed')
    count += 1
    (root / 'CMakeLists.txt').write_text('set(tag "v1.2.3")\n')
    run(root, 'git', 'add', 'CMakeLists.txt')
    p = run(root, 'git', 'commit', '-m', 'missing RELEASE rejected', ok=False)
    assert 'CMake RELEASE' in p.stdout
    count += 1
    (root / 'CMakeLists.txt').write_text(valid)
    run(root, 'git', 'add', 'CMakeLists.txt')
    p = run(root, sys.executable, '-B', str(CHECK))
    assert 'PASS' in p.stdout
    count += 1
    (root / 'src/defs.h').write_text('#define RELEASE "v0.0.0"\n')
    run(root, 'git', 'add', 'src/defs.h')
    p = run(root, 'git', 'commit', '-m', 'legacy tag mismatch rejected', ok=False)
    assert 'legacy RELEASE' in p.stdout
    count += 1
    run(root, 'git', 'reset', '--hard', 'HEAD')
    run(root, 'git', 'rm', '-q', 'CMakeLists.txt')
    p = run(root, 'git', 'commit', '-m', 'reject staged CMake deletion', ok=False)
    assert 'missing staged' in p.stdout
    count += 1
    run(root, 'git', 'reset', '--hard', 'HEAD')
    run(root, 'git', 'rm', '-q', 'src/defs.h')
    p = run(root, 'git', 'commit', '-m', 'reject staged defs deletion', ok=False)
    assert 'missing staged' in p.stdout
    count += 1
print('UPSTREAM_PRECOMMIT_REAL_GIT_HOOK_TESTS=PASS', count, 'of', count)
