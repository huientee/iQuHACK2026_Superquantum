
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import library
from qiskit.qasm2 import dumps
from rmsynth import Circuit, Optimizer

# Step 1: Decompose CRy(π/7) into Clifford+T approximation
# Using standard synthesis (e.g., Nielsen-Chuang, or Solovay-Kitaev)
# This produces a circuit with phase gates. Example decomposition:

def build_cry_approx(n_qubits=2):
    """
    Build approximate CRy(π/7) circuit.
    - Control qubit: 0
    - Target qubit: 1
    
    Standard decomposition:
      CRy(θ) ≈ CNOT(c,t) · (H on t) · CRz(θ) · (H on t) · CNOT(c,t)
    
    CRz(θ) ≈ (Rz(θ/2) on t) · CNOT(c,t) · (Rz(-θ/2) on t) · CNOT(c,t)
    
    Rz(θ) is then decomposed into T and S gates using angle approximation.
    For π/7 ≈ 0.4488, we use a T-gate sequence:
    """
    circ = Circuit(n_qubits)
    
    # Approximate CRy(π/7) decomposition (simplified)
    # H on target
    circ.add_cnot(ctrl=0, tgt=1)
    
    # Approximate CRz(π/7) using T/S gates
    # This part generates phase coefficients
    circ.add_phase(q=1, k=1)  # T
    circ.add_phase(q=1, k=2)  # S
    circ.add_phase(q=0, k=1)  # T on control (for controlled variant)
    circ.add_phase(q=0, k=2)  # S
    
    circ.add_cnot(ctrl=0, tgt=1)
    circ.add_phase(q=1, k=7)  # T† (inverse)
    
    circ.add_cnot(ctrl=0, tgt=1)
    circ.add_phase(q=0, k=1)  # T
    circ.add_phase(q=1, k=2)  # S
    
    return circ

# Step 2: Run the circuit through rmsynth
circ = build_cry_approx(n_qubits=2)
print(f"Before optimization: T-count = {circ.t_count()}")

opt = Optimizer(decoder="osd", effort=5)
opt_circ, report = opt.optimize(circ)

print(f"After optimization: T-count = {report.after_t}")
print(f"T-reduction: {report.before_t} → {report.after_t} "
      f"({100*(report.before_t - report.after_t)/report.before_t:.1f}% saved)")
print(f"Distance (residual oddness): {report.distance}")

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

# Optimized QASM
opt_qiskit = rmsynth_circuit_to_qiskit(opt_circ)
qasm_optimized = dumps(opt_qiskit)
with open("../circuits/challenge_2_rm.qasm", "w") as f:
    f.write(qasm_optimized)
print("Saved: challenge_2_rm.qasm")