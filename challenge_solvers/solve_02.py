#!/usr/bin/env python3
r"""
IQuHACK challenge 2 - Controlled-Ry(pi/7), same structure as CY (Challenge 1).

Relation to CY: CY = (I \otimes H) \cdot CZ \cdot (I \otimes H), with CZ diagonal phases [0, 0, 0, pi].
CRy(theta) = (I \otimes H) \cdot CRz(theta) \cdot (I \otimes H), with CRz(theta) diagonal [0, 0, theta/2, -theta/2].
So CRy(pi/7) has the same form as CY: H on target, then a diagonal, then H on target.
We use CRz(pi/7) diagonal [0, 0, pi/14, -pi/14]. We cannot write CRy(pi/7) as a product
of CY gates (CY fixes angle pi; we need pi/7), but the circuit structure is analogous.
We always use the real angle pi/7 when Qiskit is available (Clifford+T synthesis of CRz(pi/7)).
Without Qiskit we fall back to rmsynth on the real phases (Z8 rounding may yield identity, then we use a single Z8 term).
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
    """Convert diagonal phases into Z8 coefficient vector for rmsynth. Round 0.5 up so pi/4 works."""
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
    return [int(round(v + 1e-12)) % 8 for v in a_real]


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


def _synthesize_crz_pi7_qiskit() -> str | None:
    """CRy(pi/7) = (I⊗H) CRz(pi/7) (I⊗H). Synthesize via CRz (real angle). Return None if Qiskit unavailable."""
    try:
        from qiskit import QuantumCircuit
        from qiskit.compiler import transpile
        from qiskit.qasm2 import dumps
        qc = QuantumCircuit(2)
        qc.h(1)
        qc.crz(PI / 7, 0, 1)
        qc.h(1)
        t = transpile(qc, basis_gates=["cx", "h", "t", "tdg"], optimization_level=2)
        return dumps(t)
    except Exception:
        return None


# CRz(θ) diagonal: when control=1, Rz(θ) on target → phases 0, 0, -θ/2, θ/2 (basis |00⟩,|01⟩,|10⟩,|11⟩).
# So CRz(pi/7): phi(|00⟩)=0, phi(|01⟩)=0, phi(|10⟩)=-pi/14, phi(|11⟩)=pi/14
CRZ_PI7_PHASES = [0.0, 0.0, -PI / 14, PI / 14]


def _solve_challenge_02_rmsynth(phases: list[float], use_best_efforts: bool) -> str:
    """CRy(pi/7) = (I⊗H) CRz(pi/7) (I⊗H). Build CRz diagonal from phases, rmsynth, then H before and after."""
    vec = phase_vector_from_diagonal(phases, 2)
    if all(v == 0 for v in vec):
        vec = [0, 0, 1]
    diag_qasm = optimize_diagonal_best(vec, 2) if use_best_efforts else optimize_diagonal_and_qasm(vec, 2, effort=5)
    lines = ["OPENQASM 2.0;", "include \"qelib1.inc\";", "qreg q[2];", "h q[1];"]
    for ln in diag_qasm.split("\n"):
        if ln.strip() and "OPENQASM" not in ln and "include" not in ln and "qreg" not in ln:
            lines.append(ln)
    lines.append("h q[1];")
    return "\n".join(lines)


def solve_challenge_02(*, use_best_efforts: bool = False, use_real: bool = True, **kwargs) -> str:
    # use_real=True: Qiskit (real pi/7) when available; use_real=False: rmsynth only (Z8 approx).
    if use_real:
        qasm_qiskit = _synthesize_crz_pi7_qiskit()
        if qasm_qiskit is not None:
            return qasm_qiskit
    return _solve_challenge_02_rmsynth(CRZ_PI7_PHASES, use_best_efforts)


def main():
    out_dir = os.path.join(REPO_ROOT, "challenge_qasm")
    os.makedirs(out_dir, exist_ok=True)
    qasm = solve_challenge_02()
    path = os.path.join(out_dir, "challenge_02.qasm")
    with open(path, "w") as f:
        f.write(qasm)
    t, cx = count_costs(qasm)
    print(f"Challenge 2: {path} T={t}, CNOT={cx}")


if __name__ == "__main__":
    main()
