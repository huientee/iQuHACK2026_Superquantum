from qiskit import QuantumCircuit

qc = QuantumCircuit(2)

# SWAP q[0] <-> q[1]
qc.cx(0, 1)
qc.cx(1, 0)
qc.cx(0, 1)

print(qc)
