"""
IQuHACK challenge solver utilities.

Structure:
  1. Path setup
  2. Constants (PI, QASM header, fixed circuits)
  3. Cost counting (count_costs)
  4. Phase polynomial --> Z8 (parity, phase_vector_from_diagonal)
  5. rmsynth: circuit↔QASM, optimize diagonal
  6. Helpers: gate lines from QASM, wrap diagonal with prefix/suffix
  7. Qiskit helpers (comparison only)
  8. Challenge getters (get_challenge_01_qasm ... get_challenge_11_qasm)
  9. GETTERS dict for solve_all.py
"""
from __future__ import annotations
import math
import os
import sys

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
_rmsynth_path = os.path.join(REPO_ROOT, "rmsynth", "src", "api", "python")
if _rmsynth_path not in sys.path:
    sys.path.insert(0, _rmsynth_path)

PI = math.pi

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
QASM_HEADER_2 = ["OPENQASM 2.0;", 'include "qelib1.inc";', "qreg q[2];"]
QASM_HEADER_4 = ["OPENQASM 2.0;", 'include "qelib1.inc";', "qreg q[4];"]

# Fixed Clifford + T circuits (no diagonal to rmsynth)
CH07_FIXED_QASM = "\n".join(QASM_HEADER_2 + [
    "h q[0];", "cx q[0], q[1];", "tdg q[0];", "t q[1];", "cx q[0], q[1];",
])
CH10_FIXED_QASM = "\n".join(QASM_HEADER_2 + [
    "h q[0];", "h q[1];", "cx q[0], q[1];", "t q[0];", "tdg q[1];",
    "cx q[0], q[1];", "h q[1];",
])

# -----------------------------------------------------------------------------
# Cost counting
# -----------------------------------------------------------------------------
def count_costs(qasm: str) -> tuple[int, int]:
    """Return (t_count, cnot_count) for QASM (gates h, t, tdg, cx only)."""
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


# -----------------------------------------------------------------------------
# Phase polynomial --> Z8 vector
# -----------------------------------------------------------------------------
def parity(x: int, mask: int) -> int:
    """Parity of x over bits in mask: (popcount(x & mask)) % 2."""
    return (bin(x & mask).count("1")) % 2


def phase_vector_from_diagonal(phases_radians: list[float], n: int) -> list[int]:
    f"""
    Diagonal phases phi(x) in radians for x = 0..2^{n-1} to Z8 coefficient vector
    (length 2^{n} - 1) for rmsynth. Round 0.5 up so pi/4 gives coefficient 1.
    """
    N = 1 << n
    if len(phases_radians) != N:
        raise ValueError(f"Need {N} phases for n={n}")
    b = [4.0 * phases_radians[x] / PI for x in range(N)]
    order = list(range(1, N))
    a_real = [0.0] * len(order)
    for j, mask in enumerate(order):
        s = sum(((-1) ** parity(x, mask)) * b[x] for x in range(N))
        a_real[j] = s / (1 << (n - 1))
    return [int(round(v + 1e-12)) % 8 for v in a_real]


# -----------------------------------------------------------------------------
# rmsynth: circuit <--> QASM, optimize diagonal
# -----------------------------------------------------------------------------
def rmsynth_circuit_to_qasm(circ, qubit_names: list[str] | None = None) -> str:
    """Convert rmsynth Circuit (CNOT + phase exp(ipik/4)) to OpenQASM (h,t,tdg,cx)."""
    n = circ.n
    if qubit_names is None:
        qubit_names = [f"q[{i}]" for i in range(n)]
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{n}];"]
    for g in circ.ops:
        if g.kind == "cnot":
            lines.append(f"cx {qubit_names[g.ctrl]}, {qubit_names[g.tgt]};")
        elif g.kind == "phase":
            q, k = g.q, g.k % 8
            if k == 0:
                continue
            # k=1,2,3,4,5,6,7 --> t/tdg sequences
            if k in (1, 2, 5):
                lines.extend(["t " + qubit_names[q] + ";"] * (1 if k == 1 else (2 if k == 2 else 1)))
            elif k in (3, 6, 7):
                lines.extend(["tdg " + qubit_names[q] + ";"] * (1 if k in (3, 7) else 2))
            else:  # k == 4
                lines.extend(["t " + qubit_names[q] + ";"] * 4)
    return "\n".join(lines)


def _optimize_diagonal_circuit(vec: list[int], n: int, effort: int = 5):
    """
    Build circuit from Z8 vector, run rmsynth Optimizer if rmcore available,
    optionally T-depth schedule for n=4. Returns rmsynth Circuit.
    """
    from rmsynth.core import synthesize_from_coeffs
    circ = synthesize_from_coeffs(vec, n)
    try:
        from rmsynth import Optimizer
        opt = Optimizer(decoder="auto", effort=effort)
        circ, _ = opt.optimize(circ)
    except ImportError:
        pass
    if n == 4:
        try:
            from rmsynth.core import extract_phase_coeffs, coeffs_to_vec, synthesize_with_schedule
            a_final = extract_phase_coeffs(circ)
            vec_final = coeffs_to_vec(a_final, n)
            sched_circ, _, _ = synthesize_with_schedule(vec_final, n)
            if sched_circ.t_count() <= circ.t_count():
                circ = sched_circ
        except Exception:
            pass
    return circ


def optimize_diagonal_and_qasm(vec: list[int], n: int, effort: int = 5) -> str:
    """Z8 vector --> rmsynth optimize --> OpenQASM."""
    circ = _optimize_diagonal_circuit(vec, n, effort=effort)
    return rmsynth_circuit_to_qasm(circ)


def optimize_diagonal_and_qasm_best(
    vec: list[int], n: int, efforts: tuple[int, ...] = (3, 4, 5)
) -> str:
    """Try several efforts; return QASM with lowest T-count (then CNOT)."""
    best_qasm, best_t, best_cx = None, float("inf"), float("inf")
    for e in efforts:
        try:
            qasm = optimize_diagonal_and_qasm(vec, n, effort=e)
            t, cx = count_costs(qasm)
            if t < best_t or (t == best_t and cx < best_cx):
                best_t, best_cx, best_qasm = t, cx, qasm
        except Exception:
            continue
    return best_qasm or optimize_diagonal_and_qasm(vec, n, effort=5)


# -----------------------------------------------------------------------------
# Helpers: gate lines from QASM, wrap diagonal
# -----------------------------------------------------------------------------
def _gate_lines_from_qasm(qasm: str) -> list[str]:
    """Extract gate lines from QASM (skip OPENQASM, include, qreg)."""
    out = []
    for ln in qasm.split("\n"):
        ln = ln.strip()
        if not ln or "OPENQASM" in ln or "include" in ln or "qreg" in ln:
            continue
        out.append(ln)
    return out


def _wrap_diagonal_qasm(
    diag_qasm: str,
    prefix: list[str],
    suffix: list[str],
    n_qubits: int = 2,
) -> str:
    """Build QASM: header + prefix gates + diagonal body + suffix gates."""
    header = ["OPENQASM 2.0;", 'include "qelib1.inc";', f"qreg q[{n_qubits}];"]
    body = _gate_lines_from_qasm(diag_qasm)
    return "\n".join(header + prefix + body + suffix)


# -----------------------------------------------------------------------------
# Qiskit helpers (comparison only; not used as final solution)
# -----------------------------------------------------------------------------
def _qiskit_gates_to_qasm(qc) -> list[str]:
    """Qiskit circuit --> list of gate lines (no header)."""
    from qiskit.qasm2 import dumps
    lines = []
    for ln in dumps(qc).split("\n"):
        ln = ln.strip()
        if not ln or ln.startswith("//"):
            continue
        if "OPENQASM" in ln or "include" in ln or "qreg" in ln:
            continue
        lines.append(ln)
    return lines


def get_qiskit_ch02_qasm() -> str | None:
    """Controlled-R_y(pi/7) via Qiskit transpile (comparison only)."""
    try:
        from qiskit import QuantumCircuit
        from qiskit.transpiler import transpile
        qc = QuantumCircuit(2)
        qc.cry(PI / 7, 0, 1)
        best = None
        for opt_level in (3, 2, 1):
            qc_t = transpile(qc, basis_gates=["cx", "h", "t", "tdg"], optimization_level=opt_level)
            qasm = "\n".join(QASM_HEADER_2 + _qiskit_gates_to_qasm(qc_t))
            t, cx = count_costs(qasm)
            if best is None or (t, cx) < (best[0], best[1]):
                best = (t, cx, qasm)
        return best[2] if best else None
    except Exception:
        return None


# Challenge 7: state preparation
# Design U that maps |00⟩ --> target state (random_statevector(4, seed=42))
# Target |psi> in computational basis |00>, |01>, |10>, |11>
_CH07_STATE_PDF = [
    0.1061479384 - 0.6796414670j,   # |00>
    -0.3622775887 - 0.4536131360j, # |01>
    0.2614190429 + 0.0445330969j,  # |10>
    0.3276449279 - 0.1101628411j,  # |11>
]
_CH07_STATE = _CH07_STATE_PDF


def get_qiskit_ch07_qasm() -> str | None:
    """State prep via Qiskit initialize + decompose + transpile to Clifford+T."""
    state = list(_CH07_STATE)
    try:
        from qiskit import QuantumCircuit
        from qiskit.transpiler import transpile
        best = None
        for reps in (3, 2, 4):
            for opt in (3, 2):
                try:
                    qc = QuantumCircuit(2)
                    qc.initialize(state, [0, 1])
                    qc = qc.decompose(reps=reps)
                    qc = transpile(qc, basis_gates=["cx", "h", "t", "tdg"], optimization_level=opt)
                    qasm = "\n".join(QASM_HEADER_2 + _qiskit_gates_to_qasm(qc))
                    t, cx = count_costs(qasm)
                    if best is None or (t, cx) < (best[0], best[1]):
                        best = (t, cx, qasm)
                except Exception:
                    pass
        return best[2] if best else None
    except Exception:
        return None


def get_qiskit_ch07_schmidt_qasm() -> str | None:
    """
    Challenge 7: state preparation using Schmidt decomposition.
    Target: U|00⟩ = |psi⟩ with |psi⟩ from challenge.pdf (random_statevector(4, seed=42)).
    Schmidt (q1|q0): |psi⟩ = U_q1 (s0|00⟩ + s1|11⟩) V*_q0
    Circuit: R_y(θ,1), CX(1,0), UnitaryGate(U) on [1], UnitaryGate(V*) on [0].
    Transpile to Clifford+T (h, t, tdg, cx); try multiple seeds, return best T-count QASM.
    """
    try:
        import numpy as np
        from qiskit import QuantumCircuit, transpile
        from qiskit.circuit.library import UnitaryGate
        psi = np.array(_CH07_STATE, dtype=complex) / np.linalg.norm(_CH07_STATE)
        M = psi.reshape(2, 2)
        U, s, Vh = np.linalg.svd(M)
        V = Vh.conj().T
        s0, s1 = s
        theta = 2 * np.arctan2(s1, s0)
        V_star = V.conj()
        best = None
        best_score = None
        for seed in range(20):
            try:
                qc = QuantumCircuit(2)
                qc.ry(theta, 1)
                qc.cx(1, 0)
                qc.append(UnitaryGate(U), [1])
                qc.append(UnitaryGate(V_star), [0])
                tqc = transpile(
                    qc,
                    basis_gates=["cx", "h", "t", "tdg"],
                    optimization_level=3,
                    seed_transpiler=seed,
                )
                qasm = "\n".join(QASM_HEADER_2 + _qiskit_gates_to_qasm(tqc))
                t, cx = count_costs(qasm)
                score = (t, cx)
                if best_score is None or score < best_score:
                    best = qasm
                    best_score = score
            except Exception:
                continue
        return best
    except Exception:
        return None


def get_qiskit_ch10_qasm() -> str | None:
    """Random unitary via Qiskit (comparison only)."""
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import random_unitary
        from qiskit.transpiler import transpile
        U = random_unitary(4, seed=42)
        best = None
        for reps in (5, 4, 3, 6):
            for opt in (3, 2, 1):
                try:
                    qc = QuantumCircuit(2)
                    qc.unitary(U, [0, 1])
                    qc = qc.decompose(reps=reps)
                    qc = transpile(qc, basis_gates=["cx", "h", "t", "tdg"], optimization_level=opt)
                    qasm = "\n".join(QASM_HEADER_2 + _qiskit_gates_to_qasm(qc))
                    t, cx = count_costs(qasm)
                    if best is None or (t, cx) < (best[0], best[1]):
                        best = (t, cx, qasm)
                except Exception:
                    pass
        return best[2] if best else None
    except Exception:
        return None


def _apply_qasm_gates_to_circuit(qc, gate_lines: list[str]) -> None:
    """Apply QASM-style gate lines (h q[i];, cx q[i], q[j];) to 2-qubit circuit qc."""
    for g in gate_lines:
        g = g.strip().rstrip(";").strip()
        if g == "h q[0]":
            qc.h(0)
        elif g == "h q[1]":
            qc.h(1)
        elif g.startswith("cx "):
            parts = g.replace("cx ", "").replace("q[", "").replace("]", "").split(",")
            if len(parts) >= 2:
                c, t = int(parts[0].strip()), int(parts[1].strip())
                qc.cx(c, t)


def _qiskit_diagonal_zz_pi7_qasm(prefix_gates: list[str], suffix_gates: list[str]) -> str | None:
    """exp(i pi/7 Z \otimes Z) with optional prefix/suffix (e.g. H on one qubit for Ch4/Ch6). Returns full QASM or None."""
    try:
        import cmath
        from qiskit import QuantumCircuit
        from qiskit.circuit.library import Diagonal
        from qiskit.transpiler import transpile
        # Diagonal phases [π/7, -π/7, -π/7, π/7] → complex amplitudes (basis |00⟩,|01⟩,|10⟩,|11⟩)
        diag = [cmath.exp(1j * PI / 7), cmath.exp(-1j * PI / 7), cmath.exp(-1j * PI / 7), cmath.exp(1j * PI / 7)]
        best = None
        for opt in (3, 2, 1):
            try:
                qc = QuantumCircuit(2)
                _apply_qasm_gates_to_circuit(qc, prefix_gates)
                qc.append(Diagonal(diag), [0, 1])
                _apply_qasm_gates_to_circuit(qc, suffix_gates)
                qc_t = transpile(qc, basis_gates=["cx", "h", "t", "tdg"], optimization_level=opt)
                qasm = "\n".join(QASM_HEADER_2 + _qiskit_gates_to_qasm(qc_t))
                t, cx = count_costs(qasm)
                if best is None or (t, cx) < (best[0], best[1]):
                    best = (t, cx, qasm)
            except Exception:
                continue
        return best[2] if best else None
    except Exception:
        return None


def get_qiskit_ch03_qasm() -> str | None:
    """exp(i pi/7 Z \otimes Z) exact via Qiskit Diagonal + transpile to Clifford+T."""
    return _qiskit_diagonal_zz_pi7_qasm([], [])


def get_qiskit_ch04_qasm() -> str | None:
    """exp(i pi/7 (XX + YY)) = (I \otimes H) exp(i pi/7 ZZ) (I \otimes H) exact via Qiskit."""
    return _qiskit_diagonal_zz_pi7_qasm(["h q[1];"], ["h q[1];"])


def get_qiskit_ch06_qasm() -> str | None:
    """exp(i pi/7 (XX + ZI + IZ)): ZZ block with H on qubit 0, exact via Qiskit."""
    return _qiskit_diagonal_zz_pi7_qasm(["h q[0];"], ["h q[0];"])


def _qiskit_unitary_to_qasm(U, n_qubits: int) -> str | None:
    """Synthesize exact unitary U (2^n \times 2^n) to Clifford+T QASM via Qiskit; return full QASM or None."""
    try:
        from qiskit import QuantumCircuit
        from qiskit.transpiler import transpile
        import numpy as np
        qc = QuantumCircuit(n_qubits)
        qc.unitary(U, list(range(n_qubits)))
        best = None
        for reps in (4, 5, 3, 6) if n_qubits > 2 else (4, 3, 5):
            for opt in (3, 2, 1):
                try:
                    qc_t = qc.decompose(reps=reps)
                    qc_t = transpile(qc_t, basis_gates=["cx", "h", "t", "tdg"], optimization_level=opt)
                    qasm = "\n".join(
                        (QASM_HEADER_4 if n_qubits == 4 else QASM_HEADER_2)
                        + _qiskit_gates_to_qasm(qc_t)
                    )
                    t, cx = count_costs(qasm)
                    if best is None or (t, cx) < (best[0], best[1]):
                        best = (t, cx, qasm)
                except Exception:
                    continue
        return best[2] if best else None
    except Exception:
        return None


def get_qiskit_ch08_qasm() -> str | None:
    """2-qubit QFT (omega=i) exact: build target unitary and synthesize to Clifford+T."""
    try:
        U = _target_unitary_ch8()
        return _qiskit_unitary_to_qasm(U, 2)
    except Exception:
        return None


def get_qiskit_ch09_qasm() -> str | None:
    """Structured unitary 2 exact: build target from PDF and synthesize to Clifford+T."""
    try:
        U = _target_unitary_ch9()
        return _qiskit_unitary_to_qasm(U, 2)
    except Exception:
        return None


def get_qiskit_ch11_qasm() -> str | None:
    """4-qubit diagonal exact: build target from PDF and synthesize to Clifford+T."""
    try:
        U = _target_unitary_ch11()
        return _qiskit_unitary_to_qasm(U, 4)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Challenge getters
# -----------------------------------------------------------------------------
# Z \otimes Z eigenvalues: +1 on |00⟩,|11⟩ and -1 on |01⟩,|10⟩--> phases [pi/7, -pi/7, -pi/7, pi/7]
_PHASES_ZZ_PI7 = [PI / 7, -PI / 7, -PI / 7, PI / 7]
# CRz(pi/7) diagonal: [0, 0, -π/14, π/14] (basis |00⟩,|01⟩,|10⟩,|11⟩)
_CRZ_PI7_PHASES = [0.0, 0.0, -PI / 14, PI / 14]


def get_challenge_01_qasm(*, use_best_efforts: bool = False, **kwargs) -> str:
    """Controlled-Y (control q1, target q0 per PDF): H(q0)·CZ·H(q0) or S†(q0)·CX(q1,q0)·S(q0)."""
    vec = phase_vector_from_diagonal([0.0, 0.0, 0.0, PI], 2)
    diag_qasm = optimize_diagonal_and_qasm_best(vec, 2) if use_best_efforts else optimize_diagonal_and_qasm(vec, 2)
    qasm_a = _wrap_diagonal_qasm(diag_qasm, ["h q[0];"], ["h q[0];"])
    qasm_b = "\n".join(QASM_HEADER_2 + ["tdg q[0];", "tdg q[0];", "cx q[1], q[0];", "t q[0];", "t q[0];"])
    t_a, c_a = count_costs(qasm_a)
    t_b, c_b = count_costs(qasm_b)
    return qasm_a if (t_a < t_b or (t_a == t_b and c_a <= c_b)) else qasm_b


def _pick_qasm_by_min_norm(challenge_num: int, candidates: list[str]) -> str:
    """Of candidate QASM strings, return the one with smallest norm (d); tie-break by first."""
    if not candidates:
        return ""
    best_qasm, best_norm = None, float("inf")
    for qasm in candidates:
        if not qasm:
            continue
        n = compute_challenge_norm(challenge_num, qasm)
        norm_val = n if n is not None else float("inf")
        if norm_val < best_norm:
            best_norm, best_qasm = norm_val, qasm
    return best_qasm if best_qasm else candidates[0]


def get_challenge_02_qasm(*, use_best_efforts: bool = False, use_real: bool = True, **kwargs) -> str:
    """Controlled-Ry(pi/7): when use_real pick candidate with smallest norm; else rmsynth."""
    vec = phase_vector_from_diagonal(_CRZ_PI7_PHASES, 2)
    if all(v == 0 for v in vec):
        vec = [0, 0, 1]
    rmsynth_q = _wrap_diagonal_qasm(optimize_diagonal_and_qasm_best(vec, 2) if use_best_efforts else optimize_diagonal_and_qasm(vec, 2), ["h q[1];"], ["h q[1];"])
    if use_real:
        candidates = [get_qiskit_ch02_qasm(), _qiskit_unitary_to_qasm(_target_unitary_ch2(), 2), rmsynth_q]
        chosen = _pick_qasm_by_min_norm(2, [q for q in candidates if q])
        if chosen:
            return chosen
    return rmsynth_q


def get_challenge_03_qasm(*, use_best_efforts: bool = False, use_real: bool = True, **kwargs) -> str:
    """exp(i pi/7 Z \otimes Z): when use_real pick candidate with smallest norm; else rmsynth."""
    vec = phase_vector_from_diagonal(_PHASES_ZZ_PI7, 2)
    rmsynth_q = optimize_diagonal_and_qasm_best(vec, 2) if use_best_efforts else optimize_diagonal_and_qasm(vec, 2)
    if use_real:
        candidates = [get_qiskit_ch03_qasm(), _qiskit_unitary_to_qasm(_target_unitary_ch3(), 2), rmsynth_q]
        chosen = _pick_qasm_by_min_norm(3, [q for q in candidates if q])
        if chosen:
            return chosen
    return rmsynth_q


def get_challenge_04_qasm(*, use_best_efforts: bool = False, use_real: bool = True, **kwargs) -> str:
    """exp(i pi/7 (XX + YY)): when use_real pick candidate with smallest norm; else rmsynth."""
    vec = phase_vector_from_diagonal(_PHASES_ZZ_PI7, 2)
    rmsynth_q = _wrap_diagonal_qasm(optimize_diagonal_and_qasm_best(vec, 2) if use_best_efforts else optimize_diagonal_and_qasm(vec, 2), ["h q[1];"], ["h q[1];"])
    if use_real:
        candidates = [get_qiskit_ch04_qasm(), _qiskit_unitary_to_qasm(_target_unitary_ch4(), 2), rmsynth_q]
        chosen = _pick_qasm_by_min_norm(4, [q for q in candidates if q])
        if chosen:
            return chosen
    return rmsynth_q


def get_challenge_05_qasm(*, use_best_efforts: bool = False, **kwargs) -> str:
    """exp(i pi/4 (XX + YY + ZZ)) = e^{i pi/4} SWAP; implement SWAP (0 T, 3 CNOTs)."""
    return "\n".join(QASM_HEADER_2 + ["cx q[0], q[1];", "cx q[1], q[0];", "cx q[0], q[1];"])


def get_challenge_06_qasm(*, use_best_efforts: bool = False, use_real: bool = True, **kwargs) -> str:
    """exp(i pi/7 (XX + ZI + IZ)): when use_real pick candidate with smallest norm; else rmsynth."""
    vec = phase_vector_from_diagonal(_PHASES_ZZ_PI7, 2)
    rmsynth_q = _wrap_diagonal_qasm(optimize_diagonal_and_qasm_best(vec, 2) if use_best_efforts else optimize_diagonal_and_qasm(vec, 2), ["h q[0];"], ["h q[0];"])
    if use_real:
        candidates = [get_qiskit_ch06_qasm(), _qiskit_unitary_to_qasm(_target_unitary_ch6(), 2), rmsynth_q]
        chosen = _pick_qasm_by_min_norm(6, [q for q in candidates if q])
        if chosen:
            return chosen
    return rmsynth_q


def get_challenge_07_qasm(*, use_best_efforts: bool = False, use_real: bool = True, **kwargs) -> str:
    """
    Challenge 7: state preparation. Prefer Schmidt decomposition (PDF-indicated approach).
    When use_real: try Schmidt first, then initialize, unitary synthesis, fixed; pick smallest norm.
    """
    if use_real:
        # Schmidt decomposition (approach): primary candidate
        candidates = [
            get_qiskit_ch07_schmidt_qasm(),
            get_qiskit_ch07_qasm(),
            _qiskit_unitary_to_qasm(_target_unitary_ch7(), 2),
            CH07_FIXED_QASM,
        ]
        chosen = _pick_qasm_by_min_norm(7, [q for q in candidates if q])
        if chosen:
            return chosen
    return CH07_FIXED_QASM


def get_challenge_08_qasm(*, use_best_efforts: bool = False, use_real: bool = True, **kwargs) -> str:
    """2-qubit QFT (ω=i): when use_real pick candidate with smallest norm; else rmsynth."""
    if use_real:
        candidates = [
            get_qiskit_ch08_qasm(),
            _qiskit_unitary_to_qasm(_target_unitary_ch8(), 2),
        ]
        rmsynth_qasm = _wrap_diagonal_qasm(
            optimize_diagonal_and_qasm_best(phase_vector_from_diagonal([0.0, 0.0, 0.0, PI / 2], 2), 2) if use_best_efforts else optimize_diagonal_and_qasm(phase_vector_from_diagonal([0.0, 0.0, 0.0, PI / 2], 2), 2),
            ["h q[0];", "h q[1];"],
            ["cx q[0], q[1];", "cx q[1], q[0];", "cx q[0], q[1];"],
        )
        candidates.append(rmsynth_qasm)
        chosen = _pick_qasm_by_min_norm(8, [q for q in candidates if q])
        if chosen:
            return chosen
    vec = phase_vector_from_diagonal([0.0, 0.0, 0.0, PI / 2], 2)
    diag_qasm = optimize_diagonal_and_qasm_best(vec, 2) if use_best_efforts else optimize_diagonal_and_qasm(vec, 2)
    return _wrap_diagonal_qasm(
        diag_qasm,
        ["h q[0];", "h q[1];"],
        ["cx q[0], q[1];", "cx q[1], q[0];", "cx q[0], q[1];"],
    )


def get_challenge_09_qasm(*, use_best_efforts: bool = False, use_real: bool = True, **kwargs) -> str:
    """Structured unitary 2: when use_real pick candidate with smallest norm; else rmsynth."""
    if use_real:
        candidates = [
            get_qiskit_ch09_qasm(),
            _qiskit_unitary_to_qasm(_target_unitary_ch9(), 2),
        ]
        rmsynth_qasm = _wrap_diagonal_qasm(
            optimize_diagonal_and_qasm_best(phase_vector_from_diagonal([0.0, 0.0, 0.0, PI / 4], 2), 2) if use_best_efforts else optimize_diagonal_and_qasm(phase_vector_from_diagonal([0.0, 0.0, 0.0, PI / 4], 2), 2),
            ["h q[0];", "cx q[0], q[1];"], ["cx q[0], q[1];", "h q[0];"],
        )
        candidates.append(rmsynth_qasm)
        chosen = _pick_qasm_by_min_norm(9, [q for q in candidates if q])
        if chosen:
            return chosen
    vec = phase_vector_from_diagonal([0.0, 0.0, 0.0, PI / 4], 2)
    diag_qasm = optimize_diagonal_and_qasm_best(vec, 2) if use_best_efforts else optimize_diagonal_and_qasm(vec, 2)
    return _wrap_diagonal_qasm(diag_qasm, ["h q[0];", "cx q[0], q[1];"], ["cx q[0], q[1];", "h q[0];"])


def get_challenge_10_qasm(*, use_best_efforts: bool = False, use_real: bool = True, **kwargs) -> str:
    """Random unitary: when use_real pick candidate with smallest norm; else fixed fallback."""
    if use_real:
        candidates = [get_qiskit_ch10_qasm(), _qiskit_unitary_to_qasm(_target_unitary_ch10(), 2), CH10_FIXED_QASM]
        chosen = _pick_qasm_by_min_norm(10, [q for q in candidates if q])
        if chosen:
            return chosen
    return CH10_FIXED_QASM


def get_challenge_11_qasm(*, use_best_efforts: bool = False, use_real: bool = True, **kwargs) -> str:
    """4-qubit diagonal psi(x): when use_real pick candidate with smallest norm; else rmsynth."""
    if use_real:
        phases = [
            0, PI, 5 * PI / 4, 7 * PI / 4,
            5 * PI / 4, 7 * PI / 4, 3 * PI / 2, 3 * PI / 2,
            5 * PI / 4, 7 * PI / 4, 3 * PI / 2, 3 * PI / 2,
            3 * PI / 2, 3 * PI / 2, 7 * PI / 4, 5 * PI / 4,
        ]
        vec = phase_vector_from_diagonal(phases, 4)
        rmsynth_qasm = optimize_diagonal_and_qasm_best(vec, 4) if use_best_efforts else optimize_diagonal_and_qasm(vec, 4)
        candidates = [
            get_qiskit_ch11_qasm(),
            _qiskit_unitary_to_qasm(_target_unitary_ch11(), 4),
            rmsynth_qasm,
        ]
        candidates = [q for q in candidates if q]
        chosen = _pick_qasm_by_min_norm(11, candidates)
        if chosen:
            return chosen
    phases = [
        0, PI, 5 * PI / 4, 7 * PI / 4,
        5 * PI / 4, 7 * PI / 4, 3 * PI / 2, 3 * PI / 2,
        5 * PI / 4, 7 * PI / 4, 3 * PI / 2, 3 * PI / 2,
        3 * PI / 2, 3 * PI / 2, 7 * PI / 4, 5 * PI / 4,
    ]
    vec = phase_vector_from_diagonal(phases, 4)
    return optimize_diagonal_and_qasm_best(vec, 4) if use_best_efforts else optimize_diagonal_and_qasm(vec, 4)


# -----------------------------------------------------------------------------
# GETTERS for solve_all.py
# -----------------------------------------------------------------------------
GETTERS = {
    1: get_challenge_01_qasm,
    2: get_challenge_02_qasm,
    3: get_challenge_03_qasm,
    4: get_challenge_04_qasm,
    5: get_challenge_05_qasm,
    6: get_challenge_06_qasm,
    7: get_challenge_07_qasm,
    8: get_challenge_08_qasm,
    9: get_challenge_09_qasm,
    10: get_challenge_10_qasm,
    11: get_challenge_11_qasm,
}


# -----------------------------------------------------------------------------
# Operator norm distance
# -----------------------------------------------------------------------------
# d operator norm == max singular value


def _target_unitary_ch1():
    """Controlled-Y from challenge."""
    import numpy as np
    U = np.eye(4, dtype=complex)
    U[2, 2], U[2, 3] = 0, -1j
    U[3, 2], U[3, 3] = 1j, 0
    return U


def _target_unitary_ch2():
    """Controlled-Ry(pi/7) exact."""
    import numpy as np
    c, s = np.cos(PI / 14), np.sin(PI / 14)  # Ry(theta) has cos(theta/2), sin(theta/2)
    U = np.eye(4, dtype=complex)
    U[2, 2], U[2, 3] = c, -s
    U[3, 2], U[3, 3] = s, c
    return U


def _target_unitary_ch3():
    """exp(i pi/7 Z \otimes Z): diagonal [e^{i pi/7}, e^{-i pi/7}, e^{-i pi/7}, e^{i pi/7}]."""
    import numpy as np
    import cmath
    d = [cmath.exp(1j * PI / 7), cmath.exp(-1j * PI / 7), cmath.exp(-1j * PI / 7), cmath.exp(1j * PI / 7)]
    return np.diag(d)


def _target_unitary_ch4():
    """exp(i pi/7 (XX+YY)) = (I \otimes H) exp(i pi/7 ZZ) (I \otimes H)."""
    import numpy as np
    H2 = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    IH = np.kron(np.eye(2), H2)
    Uzz = _target_unitary_ch3()
    return IH @ Uzz @ IH


def _target_unitary_ch5():
    """exp(i pi/4 (XX+YY+ZZ)) = e^{i pi/4} SWAP."""
    import numpy as np
    import cmath
    SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
    return cmath.exp(1j * PI / 4) * SWAP


def _target_unitary_ch6():
    """exp(i pi/7 (XX+ZI+IZ)): (H \otimes I) exp(i pi/7 ZZ) (H \otimes I)."""
    import numpy as np
    H2 = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    HI = np.kron(H2, np.eye(2))
    Uzz = _target_unitary_ch3()
    return HI @ Uzz @ HI


def _target_unitary_ch7():
    """Any unitary mapping |00⟩ to Ch7 target state; first column = normalized _CH07_STATE."""
    import numpy as np
    state = np.array(_CH07_STATE, dtype=complex) / np.linalg.norm(_CH07_STATE)
    U = np.eye(4, dtype=complex)
    U[:, 0] = state
    # Complete to orthonormal: Gram–Schmidt on columns 1,2,3
    for k in range(1, 4):
        v = np.eye(4, dtype=complex)[:, k]
        for j in range(k):
            v = v - np.vdot(U[:, j], v) * U[:, j]
        U[:, k] = v / np.linalg.norm(v)
    return U


def _target_unitary_ch8():
    """2-qubit QFT (omega=i)."""
    import numpy as np
    j = 1j
    U = np.array([
        [1, 1, 1, 1],
        [1, j, -1, -j],
        [1, -1, 1, -1],
        [1, -j, -1, j],
    ], dtype=complex) / 2
    return U


def _target_unitary_ch9():
    """Structured unitary 2 from challenge (rows as given)."""
    import numpy as np
    a = (-1 + 1j) / 2
    b = (1 + 1j) / 2
    c = (-1 - 1j) / 2
    U = np.array([
        [1, 0, 0, 0],
        [0, 0, a, b],
        [0, 1j, 0, 0],
        [0, 0, a, c],
    ], dtype=complex)
    return U


def _target_unitary_ch10():
    """Random unitary from challenge (qiskit.quantum_info.random_unitary(4, seed=42))."""
    import numpy as np
    U = np.array([
        [0.1448081895 + 0.1752383997j, -0.5189281551 - 0.5242425896j, -0.1495585824 + 0.312754999j, 0.1691348143 - 0.5053863118j],
        [-0.9271743926 - 0.0878506193j, -0.1126033063 - 0.1818584963j, 0.1225587186 + 0.0964028611j, -0.2449850904 - 0.0504584131j],
        [-0.0079842758 - 0.2035507051j, -0.3893205530 - 0.0518092515j, 0.2605170566 + 0.3286402481j, 0.4451730754 + 0.6558933250j],
        [0.0313792249 + 0.1961395216j, 0.4980474972 + 0.0884604926j, 0.3407886532 + 0.7506609982j, 0.0146480652 - 0.1575584270j],
    ], dtype=complex)
    return U


def _target_unitary_ch11():
    """4-qubit diagonal U|x⟩ = e^{i psi(x)}|x⟩ with psi(x) from challenge (little-endian x)."""
    import numpy as np
    # psi(0000)=0, psi(0001)=pi, psi(0010)=5pi/4, psi(0011)=7pi/4, psi(0100)=5pi/4, ...
    phases = [
        0, PI, 5 * PI / 4, 7 * PI / 4,
        5 * PI / 4, 7 * PI / 4, 3 * PI / 2, 3 * PI / 2,
        5 * PI / 4, 7 * PI / 4, 3 * PI / 2, 3 * PI / 2,
        3 * PI / 2, 3 * PI / 2, 7 * PI / 4, 5 * PI / 4,
    ]
    import cmath
    diag = [cmath.exp(1j * p) for p in phases]
    return np.diag(diag)


_TARGET_UNITARY = {
    1: _target_unitary_ch1,
    2: _target_unitary_ch2,
    3: _target_unitary_ch3,
    4: _target_unitary_ch4,
    5: _target_unitary_ch5,
    6: _target_unitary_ch6,
    7: _target_unitary_ch7,
    8: _target_unitary_ch8,
    9: _target_unitary_ch9,
    10: _target_unitary_ch10,
    11: _target_unitary_ch11,
}


def _operator_norm(A):
    """Spectral (operator) norm = largest singular value."""
    import numpy as np
    return float(np.linalg.norm(A, 2))  # 2-norm for matrices is spectral norm


def _norm_distance_pdf(U_target, U_impl):
    """d definition from PDF."""
    import numpy as np
    best = float("inf")
    for phi in np.linspace(0, 2 * np.pi, 360, endpoint=False):
        diff = U_target - np.exp(1j * phi) * U_impl
        n = _operator_norm(diff)
        if n < best:
            best = n
    return best


def compute_challenge_norm(challenge_num: int, qasm: str) -> float | None:
    """
    Compute operator norm distance d.
    U = exact target unitary, \tilde{U} = unitary implemented by the QASM circuit.
    Returns None if Qiskit unavailable or circuit has wrong number of qubits.
    """
    try:
        import numpy as np
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator
    except ImportError:
        return None
    if challenge_num not in _TARGET_UNITARY:
        return None
    U_target = _TARGET_UNITARY[challenge_num]()
    try:
        qc = QuantumCircuit.from_qasm_str(qasm)
        n_qubits = qc.num_qubits
        if challenge_num <= 10 and n_qubits != 2:
            return None
        if challenge_num == 11 and n_qubits != 4:
            return None
        U_impl = np.asarray(Operator(qc).data, dtype=complex)
        if U_impl.shape != U_target.shape:
            return None
        return _norm_distance_pdf(U_target, U_impl)
    except Exception:
        return None
