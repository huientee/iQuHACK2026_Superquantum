#!/usr/bin/env python3
"""
Challenge 7 (alternative): State preparation using Schmidt decomposition.

Schmidt decomposition (q1 | q0): |psi⟩ = U_q1 (s0|00⟩ + s1|11⟩) V*_q0
Circuit: Ry(θ,1), CX(1,0), UnitaryGate(U) on [1], UnitaryGate(V*) on [0].
Transpile to Clifford+T (h, t, tdg, cx) and pick best T-count over seeds.

Run from challenge_solvers: python solve_ch07_schmidt.py
"""
import os
import sys

_this = os.path.dirname(os.path.abspath(__file__))
_repo = os.path.dirname(_this)
if _repo not in sys.path:
    sys.path.insert(0, _repo)
if _this not in sys.path:
    sys.path.insert(0, _this)

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, Operator, state_fidelity
from qiskit.circuit.library import UnitaryGate

from utils import (
    QASM_HEADER_2,
    _qiskit_gates_to_qasm,
    count_costs,
    compute_challenge_norm,
    _CH07_STATE,
)


def main():
    # Target state from challenge.pdf (|00>, |01>, |10>, |11>)
    psi = np.array(_CH07_STATE, dtype=complex) / np.linalg.norm(_CH07_STATE)

    # Schmidt decomposition: M = psi.reshape(2,2), M = U @ diag(s) @ Vh
    M = psi.reshape(2, 2)
    U, s, Vh = np.linalg.svd(M)
    V = Vh.conj().T
    s0, s1 = s
    theta = 2 * np.arctan2(s1, s0)
    V_star = V.conj()

    # Build unitary-only state-prep circuit (Schmidt form)
    qc = QuantumCircuit(2)
    qc.ry(theta, 1)
    qc.cx(1, 0)
    qc.append(UnitaryGate(U), [1])
    qc.append(UnitaryGate(V_star), [0])

    # Verify fidelity ≈ 1
    sv = Statevector.from_instruction(qc)
    fid = state_fidelity(sv, psi)
    print("Schmidt circuit state fidelity:", fid)

    # Transpile to Clifford+T (challenge basis); try seeds for best T-count
    basis_gates = ["cx", "h", "t", "tdg"]
    best_qasm = None
    best_score = None
    for seed in range(20):
        tqc = transpile(
            qc,
            basis_gates=basis_gates,
            optimization_level=3,
            seed_transpiler=seed,
        )
        qasm = "\n".join(QASM_HEADER_2 + _qiskit_gates_to_qasm(tqc))
        t, cx = count_costs(qasm)
        score = (t, cx)
        if best_score is None or score < best_score:
            best_score = score
            best_qasm = qasm
    if best_qasm is None:
        raise RuntimeError("Schmidt transpile failed for all seeds")

    print("Best (T-count, CNOT):", best_score)
    norm_val = compute_challenge_norm(7, best_qasm)
    print("Norm d(U, Ũ):", norm_val if norm_val is not None else "N/A")

    # Write challenge_07.qasm
    out_dir = os.path.join(_repo, "challenge_qasm")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "challenge_07.qasm")
    with open(path, "w") as f:
        f.write(best_qasm)
    print("Wrote:", path)


if __name__ == "__main__":
    main()
