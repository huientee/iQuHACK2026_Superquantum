from qiskit import *
import matplotlib.pyplot as plt

def apply_t_power(qc: QuantumCircuit, q: int, k: int):
    k %= 8
    if k == 0:
        return
    if k <= 4:
        for _ in range(k):
            qc.t(q)
    else:
        for _ in range(8 - k):
            qc.tdg(q)

def phase_on_parity(qc: QuantumCircuit, subset, k: int):
    subset = list(subset)
    if len(subset) == 0:
        return
    if len(subset) == 1:
        apply_t_power(qc, subset[0], k)
        return

    target = subset[-1]
    controls = subset[:-1]

    for c in controls:
        qc.cx(c, target)

    apply_t_power(qc, target, k)

    for c in reversed(controls):
        qc.cx(c, target)

def q11_circuit():
    qc = QuantumCircuit(4, name="Q11_diag")

    terms = {
        "0001": 6,
        "0010": 6,
        "0100": 6,
        "0111": 1,
        "1000": 6,
        "1011": 1,
        "1101": 1,
        "1110": 2,
        "1111": 3,
    }

    for mask, k in terms.items():
        subset = [i for i, b in enumerate(mask) if b == "1"]
        phase_on_parity(qc, subset, k)

    return qc

qc = q11_circuit()

# 2) graphical visualization (Matplotlib)
fig = circuit_drawer(qc, output="mpl", fold=-1)
plt.show()