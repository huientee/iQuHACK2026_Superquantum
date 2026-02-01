from qiskit import QuantumCircuit, transpile
from qiskit.circuit import library
from qiskit.qasm2 import dumps
from rmsynth import Circuit, Optimizer, extract_phase_coeffs, synthesize_from_coeffs
import numpy as np

qcirc = QuantumCircuit(2)

# CRY - pi/7, control qubit 0, target qubit 1
qcirc.cry(np.pi/7, 0, 1)

transpiled = transpile(
    qcirc,
    basis_gates=['h', 't', 'tdg', 'cx', 's', 'sdg'],
    optimization_level=3
)

# Get total t/tdg
tc = transpiled.count_ops().get('t', 0) + transpiled.count_ops().get('tdg', 0)

print(f"Total T Count: {tc}")

print(qcirc.draw())
#print(transpiled.draw())