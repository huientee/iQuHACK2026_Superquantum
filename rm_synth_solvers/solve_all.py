#!/usr/bin/env python3
"""
Solve all IQuHACK challenges and write challenge_01.qasm ... challenge_11.qasm.
Uses utils.py: rmsynth for diagonals (Ch1,3,4,6,8,9,11); Qiskit for exact pi/7 when available (Ch2,3,4,6) and for Ch7,10.
Default: real (exact pi/7 via Qiskit when installed) + rmsynth fallback; --rmsynth-only for no Qiskit.
"""
import argparse
import os

from utils import GETTERS, count_costs, compute_challenge_norm


def main():
    parser = argparse.ArgumentParser(description="Solve all challenges and write QASM to challenge_qasm/")
    parser.add_argument("--best-efforts", action="store_true", default=True,
                        help="For diagonal challenges, try rmsynth efforts 3,4,5 and keep lowest T-count (default: on)")
    parser.add_argument("--no-best-efforts", action="store_true",
                        help="Disable best-efforts (single rmsynth effort per diagonal)")
    parser.add_argument("--rmsynth-only", action="store_true",
                        help="Use only rmsynth + fixed circuits (no Qiskit): Ch2 uses Z8 approx; Ch7,10 use fixed circuits")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                        help="Output directory for QASM files (default: challenge_qasm)")
    args = parser.parse_args()

    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(this_dir)
    out_dir = args.output_dir or os.path.join(repo_root, "challenge_qasm")
    out_dir = os.path.abspath(os.path.join(repo_root, out_dir) if not os.path.isabs(out_dir) else out_dir)
    os.makedirs(out_dir, exist_ok=True)

    use_best = args.best_efforts and not args.no_best_efforts
    use_best = use_best or (os.environ.get("BEST_EFFORTS", "").lower() in ("1", "true", "yes"))
    use_real = not args.rmsynth_only
    if use_best:
        print("Using best-of-multiple-efforts for diagonal challenges (efforts 3, 4, 5)...")
    if args.rmsynth_only:
        print("Using rmsynth-only: no Qiskit (Ch2 Z8 approx; Ch7,10 fixed circuits).")
    print()
    print("Ch | path | norm | T-count | CNOT")
    print("-" * 70)
    costs = []
    for n in range(1, 12):
        solver = GETTERS[n]
        qasm = solver(use_best_efforts=use_best, use_real=use_real)
        path = os.path.join(out_dir, f"challenge_{n:02d}.qasm")
        with open(path, "w") as f:
            f.write(qasm)
        t, cx = count_costs(qasm)
        norm_val = compute_challenge_norm(n, qasm)
        costs.append((n, t, cx, norm_val))
        if norm_val is not None:
            norm_str = "0.000000" if norm_val < 1e-10 else f"{norm_val:.6e}"
        else:
            norm_str = "N/A"
        print(f"Challenge {n:2d}: {path}  norm={norm_str}  T={t}  CNOT={cx}")

    total_t = sum(c[1] for c in costs)
    total_cx = sum(c[2] for c in costs)
    print()
    print(f"Total: T-count={total_t}, CNOT={total_cx}")
    print(f"Output: {out_dir}/challenge_01.qasm ... challenge_11.qasm")
    print("Submit at iquhack.superquantum.io")


if __name__ == "__main__":
    main()
