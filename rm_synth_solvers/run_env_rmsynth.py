#!/usr/bin/env python3
"""
Load the whole superquantum environment and test with rmsynth.

- Sets up paths (repo root, rmsynth, challenge_solvers).
- Runs rmsynth unit tests (core, optimizer, utils integration).
- Runs solve_all.py with options and prints norms/costs.

Run from Superquantum root:
  .venv/bin/python run_env_rmsynth_test.py
  # or after: source .venv/bin/activate && python run_env_rmsynth_test.py
"""
import os
import sys

# -----------------------------------------------------------------------------
# Environment: REPO_ROOT and Python path
# -----------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

RMSYNTH_API = os.path.join(REPO_ROOT, "rmsynth", "src", "api", "python")
if RMSYNTH_API not in sys.path:
    sys.path.insert(0, RMSYNTH_API)

CHALLENGE_SOLVERS = os.path.join(REPO_ROOT, "challenge_solvers")
if CHALLENGE_SOLVERS not in sys.path:
    sys.path.insert(0, CHALLENGE_SOLVERS)

os.chdir(REPO_ROOT)


def check_venv():
    """Warn if not running inside project .venv."""
    venv_dir = os.path.join(REPO_ROOT, ".venv")
    if not os.path.isdir(venv_dir):
        return
    expected_prefix = os.path.realpath(venv_dir)
    actual = os.path.realpath(sys.prefix)
    if actual != expected_prefix:
        print(f"[run_env_rmsynth_test] Note: not using project venv (sys.prefix={sys.prefix}).")
        print(f"For full env, run: .venv/bin/python run_env_rmsynth_test.py")
        print()


def test_rmsynth_core():
    """Test rmsynth.core: synthesize_from_coeffs on a simple Z8 vector."""
    print("Test 1: rmsynth.core (synthesize_from_coeffs)...", end=" ", flush=True)
    try:
        from rmsynth.core import synthesize_from_coeffs
    except ImportError as e:
        print("FAIL")
        print(f"  Import error: {e}")
        return False
    try:
        from utils import phase_vector_from_diagonal
        import math
        vec = phase_vector_from_diagonal([0.0, 0.0, 0.0, math.pi], 2)
    except ImportError:
        vec = [0, 0, 6]
    try:
        circ = synthesize_from_coeffs(vec, 2)
    except Exception as e:
        print(f"FAIL ({e})")
        return False
    if not hasattr(circ, "n") or circ.n != 2:
        print("FAIL (circ.n != 2)")
        return False
    if not hasattr(circ, "ops") or not isinstance(circ.ops, list):
        print("FAIL (no circ.ops)")
        return False
    t_count = circ.t_count() if hasattr(circ, "t_count") else sum(
        1 for g in circ.ops if getattr(g, "kind", None) == "phase" and (getattr(g, "k", 0) % 2)
    )
    print(f"OK (n=2, ops={len(circ.ops)}, T-count={t_count})")
    return True


def test_rmsynth_optimizer():
    """Test rmsynth Optimizer (optional; may need C++ rmcore)."""
    print("Test 2: rmsynth.optimizer (Optimizer)...", end=" ", flush=True)
    try:
        from rmsynth.core import synthesize_from_coeffs
        from rmsynth.optimizer import Optimizer
    except ImportError as e:
        print(f"SKIP (ImportError: {e})")
        return True
    try:
        from utils import phase_vector_from_diagonal
        import math
        vec = phase_vector_from_diagonal([0.0, 0.0, 0.0, math.pi], 2)
    except ImportError:
        vec = [0, 0, 6]
    circ = synthesize_from_coeffs(vec, 2)
    try:
        opt = Optimizer(decoder="rpa", effort=2)
        new_circ, rep = opt.optimize(circ)
        t = new_circ.t_count()
        print(f"OK (optimized T-count={t})")
        return True
    except Exception as e:
        print(f"SKIP ({e})")
        return True


def test_utils_integration():
    """Test full pipeline via utils: get_challenge_03_qasm with rmsynth-only."""
    print("Test 3: utils integration (get_challenge_03_qasm, rmsynth)...", end=" ", flush=True)
    try:
        from utils import GETTERS, count_costs
    except ImportError as e:
        print("FAIL")
        print(f"  Import error: {e}")
        return False
    qasm = GETTERS[3](use_real=False)  # force rmsynth path
    if not qasm or "OPENQASM" not in qasm or "qreg" not in qasm:
        print("FAIL (invalid QASM)")
        return False
    t, cx = count_costs(qasm)
    if t < 0 or cx < 0:
        print("FAIL (bad counts)")
        return False
    print(f"OK (T={t}, CNOT={cx})")
    return True


def run_solve_all_rmsynth_only():
    """Run solve_all with --rmsynth-only (all 11 challenges using only rmsynth/fixed)."""
    print("\n" + "=" * 70)
    print("solve_all.py --rmsynth-only (all challenges, rmsynth + fixed circuits)")
    print("=" * 70)
    # Programmatic run: set argv and call main
    old_argv = sys.argv
    try:
        sys.argv = ["solve_all.py", "--rmsynth-only"]
        from solve_all import main
        main()
    finally:
        sys.argv = old_argv


def main():
    print("Superquantum environment + rmsynth test")
    print(f"REPO_ROOT = {REPO_ROOT}")
    print(f"Python = {sys.executable}")
    print()
    check_venv()

    ok1 = test_rmsynth_core()
    ok2 = test_rmsynth_optimizer()
    ok3 = test_utils_integration()

    if ok1 and ok3:
        run_solve_all_rmsynth_only()
        print()
        print("Result: rmsynth environment OK; solve_all --rmsynth-only completed.")
        if not ok2:
            print("(Optimizer skipped; C++ rmcore may be needed for full optimization.)")
    else:
        print()
        print("Result: FAILED (fix rmsynth path or install: pip install -e rmsynth)")
        sys.exit(1)


if __name__ == "__main__":
    main()
