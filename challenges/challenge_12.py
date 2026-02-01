"""
Challenge 12: json parser
"""
import json
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import library
from qiskit.qasm2 import dumps
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate, HamiltonianGate, UnitaryGate
from rmsynth import Circuit, Optimizer, extract_phase_coeffs, synthesize_from_coeffs
import numpy as np
from scipy.linalg import expm

def invert_cliffords(cliffords):
    inv = []
    for gate, q in reversed(cliffords):
        if gate == "H":
            inv.append(("H", q))
        elif gate == "Tdg":
            inv.append(("T", q))
        elif gate == "T":
            inv.append(("Tdg", q))
    return inv

def pauli_to_z_mask(pauli):
    """
    Returns:
      mask: tuple[int] length n, 1 if Pauli != I
      cliffords: list of (gate, qubit) using ONLY H, T, Tdg
    """
    mask = []
    cliffords = []

    for i, p in enumerate(pauli):
        if p == "I":
            mask.append(0)

        elif p == "Z":
            mask.append(1)

        elif p == "X":
            mask.append(1)
            cliffords.append(("H", i))

        elif p == "Y":
            mask.append(1)
            cliffords.append(("Tdg", i))
            cliffords.append(("Tdg", i))
            cliffords.append(("H", i))

        else:
            raise ValueError(p)

    return tuple(mask), cliffords


def parse_challenge12(filename="challenge_12.json"):
    with open(filename, "r") as f:
        data = json.load(f)

    gate_strings = []
    k_values = []

    for term in data["terms"]:
        pauli = term["pauli"]          # length-9 string
        k = term["k"]

        # store exactly 9 at a time (one gate string per term)
        gate_strings.append(pauli)
        k_values.append(k)

    return gate_strings, k_values

def build_phase_polynomial(gates, ks):
    """
    Returns:
      poly: dict[mask -> coeff mod 8]
      cliffords: list of basis-change ops
    """
    poly = {}
    cliffords = []

    for pauli, k in zip(gates, ks):
        mask, ops = pauli_to_z_mask(pauli)
        cliffords.extend(ops)
        poly[mask] = (poly.get(mask, 0) + k) % 8

    return poly, cliffords

def main():
    # Helper function to convert rmsynth to Qiskit
    def rmsynth_circuit_to_qiskit(rmsynth_circ):
        from qiskit import QuantumCircuit
        qirc = QuantumCircuit(rmsynth_circ.n)
        for gate in rmsynth_circ.ops:
            if gate.kind == "cnot":
                qirc.cx(gate.ctrl, gate.tgt)
            elif gate.kind == "phase":
                k = gate.k % 8
                if k == 1:
                    qirc.t(gate.q)
                elif k == 7:
                    qirc.tdg(gate.q)
                elif k == 2:
                    qirc.s(gate.q)
                elif k == 6:
                    qirc.sdg(gate.q)
                elif k == 4:
                    # Z = S @ S, avoid using Z gate directly
                    qirc.s(gate.q)
                    qirc.s(gate.q)
                # k=0, 3, 5 are identity or unnecessary
        return qirc
    
    gates, ks = parse_challenge12()
    n = 9

    poly, cliffords = build_phase_polynomial(gates, ks)

    circ = Circuit(n)

    # build parity gadgets
    for mask, k in poly.items():
        if k % 8 == 0:
            continue

        qubits = [i for i, b in enumerate(mask) if b]
        if not qubits:
            continue

        tgt = qubits[0]
        for q in qubits[1:]:
            circ.add_cnot(q, tgt)

        circ.add_phase(tgt, k)

        for q in reversed(qubits[1:]):
            circ.add_cnot(q, tgt)

    # optimize
    opt = Optimizer(decoder="rpa", effort=5)
    opt_circ, report = opt.optimize(circ)

    print("Optimized T-count:", report.after_t)

    # convert back to qiskit
    qiskit_phase = rmsynth_circuit_to_qiskit(opt_circ)

    # add Clifford layers
    qcirc = QuantumCircuit(n)

    for gate, q in cliffords:
        if gate == "H":
            qcirc.h(q)
        elif gate == "Tdg":
            qcirc.tdg(q)
        elif gate == "T":
            qcirc.t(q)
    qcirc.compose(qiskit_phase, inplace=True)

    inv = invert_cliffords(cliffords)

    for gate, q in inv:
        if gate == "H":
            qcirc.h(q)
        elif gate == "Tdg":
            qcirc.tdg(q)
        elif gate == "T":
            qcirc.t(q)

    qasm = dumps(qcirc)

    with open("../circuits/challenge_12_synth.qasm", "w") as f:
        f.write(qasm)

if __name__ == "__main__":
    main()