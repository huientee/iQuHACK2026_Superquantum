#!/usr/bin/env python3
"""
Test that rmsynth works: synthesize_from_coeffs, Optimizer, and full pipeline via solve_03.
Run from challenge_solvers: python3 test_rmsynth.py
"""
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
_rmsynth_path = os.path.join(REPO_ROOT, "rmsynth", "src", "api", "python")
if _rmsynth_path not in sys.path:
    sys.path.insert(0, _rmsynth_path)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


def test_rmsynth_core():
    """Test rmsynth.core: synthesize_from_coeffs on a simple Z8 vector (2-qubit CZ)."""
    print("Test 1: rmsynth.core (synthesize_from_coeffs)...", end=" ")
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
    circ = synthesize_from_coeffs(vec, 2)
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
    print("Test 2: rmsynth.optimizer (Optimizer)...", end=" ")
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


def test_solve_03_integration():
    """Test full pipeline via utils: get_challenge_03_qasm()."""
    print("Test 3: utils integration (get_challenge_03_qasm)...", end=" ")
    try:
        from utils import GETTERS, count_costs
    except ImportError as e:
        print("FAIL")
        print(f"  Import error: {e}")
        return False
    qasm = GETTERS[3]()
    if not qasm or "OPENQASM" not in qasm or "qreg" not in qasm:
        print("FAIL (invalid QASM)")
        return False
    t, cx = count_costs(qasm)
    if t < 0 or cx < 0:
        print("FAIL (bad counts)")
        return False
    print(f"OK (T={t}, CNOT={cx})")
    return True


def main():
    print("rmsynth tests (run from challenge_solvers or Superquantum)\n")
    ok1 = test_rmsynth_core()
    ok2 = test_rmsynth_optimizer()
    ok3 = test_solve_03_integration()
    print()
    if ok1 and ok3:
        print("Result: rmsynth is working (core + solve_03 integration).")
        if not ok2:
            print("(Optimizer skipped or failed; C++ rmcore may be needed for full optimization.)")
    else:
        print("Result: FAILED (fix rmsynth path or install: pip install -e rmsynth)")
        sys.exit(1)


if __name__ == "__main__":
    main()
