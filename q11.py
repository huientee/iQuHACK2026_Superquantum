from qiskit import QuantumCircuit
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt

def tpower(qc, q, k):
    k %= 8
    if k == 2: qc.s(q); return 0
    if k == 4: qc.s(q); qc.s(q); return 0
    if k == 6: qc.sdg(q); return 0
    if k == 1: qc.t(q); return 1
    if k == 3: qc.s(q); qc.s(q); qc.tdg(q); return 1
    if k == 5: qc.s(q); qc.s(q); qc.t(q); return 1
    if k == 7: qc.tdg(q); return 1
    return 0

def parity_phase(qc, subset, k):
    subset = list(subset)
    if len(subset) == 1:
        return tpower(qc, subset[0], k)
    tgt, ctrls = subset[-1], subset[:-1]
    for c in ctrls: qc.cx(c, tgt)
    used = tpower(qc, tgt, k)
    for c in reversed(ctrls): qc.cx(c, tgt)
    return used

terms = {
    "0001": 6, "0010": 6, "0100": 6,
    "0111": 1, "1000": 6, "1011": 1,
    "1101": 1, "1110": 2, "1111": 3
}

qc = QuantumCircuit(4, name="Q11")
estT = 0
for mask, k in terms.items():
    subset = [i for i, b in enumerate(mask) if b == "1"]
    estT += parity_phase(qc, subset, k)

circuit_drawer(qc, output="mpl", fold=-1)
plt.show()

tcount = qc.count_ops().get("t", 0) + qc.count_ops().get("tdg", 0)
print("T-count =", tcount)
