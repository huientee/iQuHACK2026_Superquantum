from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.synthesis import synth_clifford_depth_lnn
from qiskit.quantum_info import random_unitary
from qiskit.circuit.library import UnitaryGate
from qiskit import transpile
import numpy as np

qcirc = QuantumCircuit(2)

# CRY - pi/7, control qubit 0, target qubit 1
qcirc.cry(np.pi/7, 0, 1)

transpiled = transpile(
    qcirc,
    basis_gates=['h', 't', 'tdg', 'cx', 's', 'sdg'],
    optimization_level=1
)

# Get total t/tdg
tc = transpiled.count_ops().get('t', 0) + transpiled.count_ops().get('tdg', 0)

print(f"Total T Count: {tc}")

print(qcirc.draw())
#print(transpiled.draw())