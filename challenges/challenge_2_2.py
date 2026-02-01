#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 18:44:46 2026

@author: teehuien
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler.passes.synthesis.plugin import unitary_synthesis_plugin_names
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import library
from qiskit.qasm2 import dumps
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import HamiltonianGate, UnitaryGate
from rmsynth import Circuit, Optimizer, extract_phase_coeffs, synthesize_from_coeffs
import numpy as np
from qiskit.qasm2 import dumps

def qiskit_circuit_to_rmsynth(qiskit_circ):
    """
    Convert a Qiskit circuit (in Clifford+T basis) to rmsynth Circuit format.
    
    Supported gates:
    - h: Hadamard (Clifford, handled implicitly)
    - cx/cnot: CNOT
    - t: T gate (phase k=1, exp(iπ/4))
    - tdg: T† gate (phase k=7, exp(-iπ/4))
    - s: S gate (phase k=2, exp(iπ/2))
    - sdg: S† gate (phase k=6, exp(-iπ/2))
    - z: Z gate (phase k=4, exp(iπ))
    """
    n = qiskit_circ.num_qubits
    rmsynth_circ = Circuit(n)
    
    for instr in qiskit_circ.data:
        gate_name = instr.operation.name.lower()
        qubits = instr.qubits
        
        if gate_name == "cx" or gate_name == "cnot":
            ctrl = qiskit_circ.find_bit(qubits[0])[0]
            tgt = qiskit_circ.find_bit(qubits[1])[0]
            rmsynth_circ.add_cnot(ctrl, tgt)
        
        elif gate_name == "t":
            q = qiskit_circ.find_bit(qubits[0])[0]
            rmsynth_circ.add_phase(q, k=1)  # exp(iπ/4)
        
        elif gate_name == "tdg":
            q = qiskit_circ.find_bit(qubits[0])[0]
            rmsynth_circ.add_phase(q, k=7)  # exp(-iπ/4) ≡ k=7 (mod 8)
        
        elif gate_name == "s":
            q = qiskit_circ.find_bit(qubits[0])[0]
            rmsynth_circ.add_phase(q, k=2)  # exp(iπ/2)
        
        elif gate_name == "sdg":
            q = qiskit_circ.find_bit(qubits[0])[0]
            rmsynth_circ.add_phase(q, k=6)  # exp(-iπ/2) ≡ k=6 (mod 8)
        
        elif gate_name == "z":
            q = qiskit_circ.find_bit(qubits[0])[0]
            rmsynth_circ.add_phase(q, k=4)  # exp(iπ) = -1
        
        elif gate_name == "h":
            # Hadamard is Clifford; handled implicitly in rmsynth
            # (CNOTs effectively implement conjugations by H)
            pass
        
        elif gate_name == "barrier":
            # Barriers don't affect phase polynomial
            pass
        
        else:
            # Ignore other gates or comment them
            print(f"Warning: Ignoring gate {gate_name}")
    
    return rmsynth_circ

theta = np.pi / 7

# epsilon = 0.22390000000000287
epsilon = 0.1958
# epilon = 0.1
# epsilon = 0.04315

qc = QuantumCircuit(2)
qc.cry(theta, 0, 1)

print("Installed synthesis plugins:", unitary_synthesis_plugin_names())

min = 1000
min_eps = 0
# t = True
t = False
qasm_str = ""
if t:
    for epsilon in np.arange(epsilon - 0.00015, 0.04, -0.00005):
        optq = transpile(
            qc,
            basis_gates=["h", "t", "tdg", "cx", "s", "sdg"],
            optimization_level=3,
            unitary_synthesis_method="gridsynth",                 # <-- Ross–Selinger
            unitary_synthesis_plugin_config={"epsilon": epsilon}  # supported by plugin
        )

        qasm_str = dumps(optq)
        #print(qasm_str)
        tc = optq.count_ops().get("t", 0) + optq.count_ops().get("tdg", 0)
        print("T-count =", tc, "Epsilon: ", epsilon)

        if tc <= min:
            min = tc
            min_eps = epsilon

        print(min, min_eps)
else :
    optq = transpile(
        qc,
        basis_gates=["h", "t", "tdg", "cx", "s", "sdg"],
        optimization_level=3,
        unitary_synthesis_method="gridsynth",                 # <-- Ross–Selinger
        unitary_synthesis_plugin_config={"epsilon": epsilon}  # supported by plugin
    )

    qasm_str = dumps(optq)
    #print(qasm_str)
    tc = optq.count_ops().get("t", 0) + optq.count_ops().get("tdg", 0)
    print("T-count =", tc, "Epsilon: ", epsilon)

with open("../circuits/challenge_2_q.qasm", "w") as f:
    f.write(qasm_str)

# print("Saved: challenge_2_q.qasm")
# rmsynth_circ = qiskit_circuit_to_rmsynth(optq)

# # Optimize with rmsynth
# print("Optimizing with rmsynth")
# opt = Optimizer(decoder="osd", effort=5)
# opt_circ = opt.optimize(rmsynth_circ)

# # Helper function to convert rmsynth to Qiskit
# def rmsynth_circuit_to_qiskit(rmsynth_circ):
#     from qiskit import QuantumCircuit
#     qirc = QuantumCircuit(rmsynth_circ.n)
#     for gate in rmsynth_circ.ops:
#         if gate.kind == "cnot":
#             qirc.cx(gate.ctrl, gate.tgt)
#         elif gate.kind == "phase":
#             k = gate.k % 8
#             if k == 1:
#                 qirc.t(gate.q)
#             elif k == 7:
#                 qirc.tdg(gate.q)
#             elif k == 2:
#                 qirc.s(gate.q)
#             elif k == 6:
#                 qirc.sdg(gate.q)
#             elif k == 4:
#                 # Z = S @ S, avoid using Z gate directly
#                 qirc.s(gate.q)
#                 qirc.s(gate.q)
#             # k=0, 3, 5 are identity or unnecessary
#     return qirc

# # Optimized QASM
# opt_qiskit = rmsynth_circuit_to_qiskit(opt_circ)
# qasm_optimized = dumps(opt_qiskit)
# with open("../circuits/challenge_2_n.qasm", "w") as f:
#     f.write(qasm_optimized)
# print("Saved: challenge_2_n.qasm")