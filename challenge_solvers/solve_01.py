#!/usr/bin/env python3
"""
IQuHACK challenge 1 - Controlled-Y

Controlled-Y can be built as Hadamard on the target, then CZ, then Hadamard again.
CZ only changes the phase of the state where both qubits are 1; it does not mix basis states,
so it is "diagonal" and we can feed it to rmsynth to get a low T-count circuit. We also try
the alternative decomposition (S-dagger, CNOT, S on target) and keep whichever has fewer T gates.
"""
import math
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "rmsynth", "src", "api", "python")):
    if p not in sys.path:
        sys.path.insert(0, p)

PI = math.pi


def count_costs(qasm: str) -> tuple[int, int]:
    t_count = cnot_count = 0
    for line in qasm.split("\n"):
        s = line.strip().split("//")[0].strip().rstrip(";").strip().lower()
        if not s or "qreg" in s or "include" in s or "openqasm" in s:
            continue
        if s.startswith("t ") or " t " in s:
            t_count += 1
        elif s.startswith("tdg ") or " tdg " in s:
            t_count += 1
        elif s.startswith("cx ") or " cx " in s:
            cnot_count += 1
    return t_count, cnot_count


def parity(x: int, mask: int) -> int:
    return (bin(x & mask).count("1")) % 2


def phase_vector_from_diagonal(phases_radians: list[float], n: int) -> list[int]:
    """Turn the list of phases (one per basis state) into the coefficient list rmsynth wants."""
    N = 1 << n
    if len(phases_radians) != N:
        raise ValueError(f"Need {N} phases for n={n}")
    b = [4.0 * phases_radians[x] / PI for x in range(N)]
    order = list(range(1, N))
    a_real = [0.0] * len(order)
    for j, mask in enumerate(order):
        s = 0.0
        for x in range(N):
            s += ((-1) ** parity(x, mask)) * b[x]
        a_real[j] = s / (1 << (n - 1))
    return [int(round(v)) % 8 for v in a_real]


def rmsynth_circuit_to_qasm(circ, qubit_names=None) -> str:
    n = circ.n
    if qubit_names is None:
        qubit_names = [f"q[{i}]" for i in range(n)]
    lines = ["OPENQASM 2.0;", "include \"qelib1.inc\";", f"qreg q[{n}];"]
    for g in circ.ops:
        if g.kind == "cnot":
            lines.append(f"cx {qubit_names[g.ctrl]}, {qubit_names[g.tgt]};")
        elif g.kind == "phase":
            q, k = g.q, g.k % 8
            if k == 0:
                continue
            if k == 1:
                lines.append(f"t {qubit_names[q]};")
            elif k == 2:
                lines.append(f"t {qubit_names[q]};")
                lines.append(f"t {qubit_names[q]};")
            elif k == 3:
                lines.append(f"tdg {qubit_names[q]};")
            elif k == 4:
                for _ in range(4):
                    lines.append(f"t {qubit_names[q]};")
            elif k == 5:
                lines.append(f"t {qubit_names[q]};")
            elif k == 6:
                lines.append(f"tdg {qubit_names[q]};")
                lines.append(f"tdg {qubit_names[q]};")
            elif k == 7:
                lines.append(f"tdg {qubit_names[q]};")
    return "\n".join(lines)


def optimize_diagonal_and_qasm(vec: list[int], n: int, effort: int = 5) -> str:
    """Build circuit from Z8 vector, run rmsynth optimizer (if rmcore available), return QASM."""
    from rmsynth.core import synthesize_from_coeffs
    circ = synthesize_from_coeffs(vec, n)
    try:
        from rmsynth import Optimizer
        opt = Optimizer(decoder="auto", effort=effort)
        circ, _ = opt.optimize(circ)
    except ImportError:
        pass
    return rmsynth_circuit_to_qasm(circ)


def optimize_diagonal_best(vec: list[int], n: int, efforts: tuple[int, ...] = (3, 4, 5)) -> str:
    """Run rmsynth at different effort levels; keep the circuit that has the fewest T gates (then CNOT)."""
    best_qasm, best_t, best_cx = None, float("inf"), float("inf")
    for e in efforts:
        try:
            qasm = optimize_diagonal_and_qasm(vec, n, effort=e)
            t, cx = count_costs(qasm)
            if t < best_t or (t == best_t and cx < best_cx):
                best_t, best_cx, best_qasm = t, cx, qasm
        except Exception:
            continue
    return best_qasm if best_qasm is not None else optimize_diagonal_and_qasm(vec, n, effort=5)


def solve_challenge_01(*, use_best_efforts: bool = False, **kwargs) -> str:
    # CZ: phase pi only when both qubits are 1; other states unchanged.
    phases_cz = [0.0, 0.0, 0.0, PI]
    vec = phase_vector_from_diagonal(phases_cz, 2)
    diag_qasm = optimize_diagonal_best(vec, 2) if use_best_efforts else optimize_diagonal_and_qasm(vec, 2, effort=5)
    # Wrap the diagonal (CZ) with Hadamard on qubit 1 to get Controlled-Y.
    lines = ["OPENQASM 2.0;", "include \"qelib1.inc\";", "qreg q[2];", "h q[1];"]
    for ln in diag_qasm.split("\n"):
        if ln.strip() and "OPENQASM" not in ln and "include" not in ln and "qreg" not in ln:
            lines.append(ln)
    lines.append("h q[1];")
    qasm_a = "\n".join(lines)
    # Alternative: S-dagger on target, CNOT, S on target; compare and keep the one with fewer T gates.
    qasm_b = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
tdg q[1];
tdg q[1];
cx q[0], q[1];
t q[1];
t q[1];
"""
    t_a, c_a = count_costs(qasm_a)
    t_b, c_b = count_costs(qasm_b)
    return qasm_a if (t_a < t_b or (t_a == t_b and c_a <= c_b)) else qasm_b


def main():
    out_dir = os.path.join(REPO_ROOT, "challenge_qasm")
    os.makedirs(out_dir, exist_ok=True)
    qasm = solve_challenge_01()
    path = os.path.join(out_dir, "challenge_01.qasm")
    with open(path, "w") as f:
        f.write(qasm)
    t, cx = count_costs(qasm)
    print(f"Challenge 1: {path} T={t}, CNOT={cx}")


if __name__ == "__main__":
    main()
