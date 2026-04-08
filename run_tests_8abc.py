"""
run_tests_8abc.py
=================
Author: Ivelin Likov

Runs Tests 8A, 8B, 8C sequentially.
Tests cross-cube composition mechanisms in SAGE.

Usage:
    python run_tests_8abc.py

Each test saves JSON + PNG in the same folder.
Estimated total: ~30-60 minutes on RTX 4090.
"""

import subprocess, sys, os
from datetime import datetime

tests = [
    ('test_8a_query_chaining.py',    'Test 8A — Residual Query Chaining    (~10 min)'),
    ('test_8b_shared_coords.py',     'Test 8B — Shared Grid Coordinates    (~15 min)'),
    ('test_8c_parallel_compose.py',  'Test 8C — Parallel Composition       (~15 min)'),
]

script_dir = os.path.dirname(os.path.abspath(__file__))
start_time = datetime.now()

print(f"\n{'█'*60}")
print(f"  SAGE Tests 8A → 8C — Cross-Cube Composition")
print(f"  Started:   {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Estimated: ~40 minutes on RTX 4090")
print(f"  Results:   {script_dir}")
print(f"{'█'*60}\n")

passed, failed = [], []

for i, (script, label) in enumerate(tests, 1):
    path = os.path.join(script_dir, script)

    if not os.path.exists(path):
        print(f"\n  [{i}/{len(tests)}] SKIP — not found: {script}\n")
        failed.append(script)
        continue

    print(f"\n{'='*60}")
    print(f"  [{i}/{len(tests)}] {label}")
    print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")

    t0     = datetime.now()
    result = subprocess.run([sys.executable, path], cwd=script_dir)
    elapsed = (datetime.now() - t0).total_seconds()
    m = int(elapsed // 60)
    s = int(elapsed % 60)

    if result.returncode != 0:
        print(f"\n  ✗ FAILED (code={result.returncode}, {m}m{s}s) — continuing\n")
        failed.append(script)
    else:
        print(f"\n  ✓ DONE ({m}m{s}s)\n")
        passed.append(script)

end_time = datetime.now()
elapsed  = end_time - start_time
h = int(elapsed.total_seconds() // 3600)
m = int((elapsed.total_seconds() % 3600) // 60)

print(f"\n{'█'*60}")
print(f"  ALL TESTS COMPLETE")
print(f"  Total time: {h}h {m}m")
print(f"  Finished:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Passed: {len(passed)}/{len(tests)}")
if failed:
    print(f"  Failed: {', '.join(failed)}")
print(f"\n  Look for results in: {script_dir}")
print(f"    test_8a_chaining_results.json / .png")
print(f"    test_8b_shared_coords_results.json / .png")
print(f"    test_8c_parallel_results.json / .png")
print(f"{'█'*60}\n")
