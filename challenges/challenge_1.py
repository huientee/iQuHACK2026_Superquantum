from qiskit import *
import numpy as np

qcirc = QuantumCircuit(2)

# CY - control qubit 0, target qubit 1
qcirc.cy(0, 1)

transpiled = transpile(
    qcirc,
    basis_gates=['h', 't', 'tdg', 'cx', 's', 'sdg'],
    optimization_level=1
)

# Get total t/tdg
tc = transpiled.count_ops().get('t', 0) + transpiled.count_ops().get('tdg', 0)

print(f"Total T Count: {tc}")

print(qcirc.draw())
print(transpiled.draw())