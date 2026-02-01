"""
Challenge 3: exp(i*pi/7 Z tens Z)
"""
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import library
from qiskit.qasm2 import dumps
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate, HamiltonianGate, UnitaryGate
from rmsynth import Circuit, Optimizer, extract_phase_coeffs, synthesize_from_coeffs
import numpy as np
from scipy.linalg import expm

def qiskit_circuit_to_rmsynth(qiskit_circ):
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

def main():
    # Define Hamiltonian H and time t
    H = np.array([[2, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, -2]])
    theta = np.pi / 7
    matrix = expm(1j * theta * H)

    # Create UnitaryGate
    unitary_gate = UnitaryGate(matrix, label="exp(i H theta)")

    # Add to circuit
    qcirc = QuantumCircuit(2)
    qcirc.append(unitary_gate, [0, 1])
    
    print("Original circuit:")
    print(qcirc.draw())
    
    # Transpile
    print("Transpiling to Clifford+T basis...")
    transpiled = transpile(
        qcirc,
        basis_gates=['h', 't', 'tdg', 'cx', 's', 'sdg'],
        optimization_level=3,
        unitary_synthesis_method="gridsynth"
    )
    
    # print("Transpiled circuit:")
    # print(transpiled.draw())
    
    # Count T-gates
    t_before = transpiled.count_ops().get('t', 0) + transpiled.count_ops().get('tdg', 0)
    print("T-count: ", t_before)
    
    qasm_baseline = dumps(transpiled)
    with open("../circuits/challenge_6_real.qasm", "w") as f:
        f.write(qasm_baseline)
    print("  Saved: challenge_6_real.qasm")

    rmsynth_circ = qiskit_circuit_to_rmsynth(transpiled)
    
    # Optimize with rmsynth
    print("Optimizing with rmsynth")
    opt = Optimizer(decoder="osd", effort=5)
    opt_circ, report = opt.optimize(rmsynth_circ)
    
    print(f"\nQiskit transpiled (baseline):")
    print(f"  T-count: {t_before}")
    
    print(f"\nrmsynth optimized:")
    print(f"  T-count: {report.after_t}")
    
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
    
    # Optimized QASM
    opt_qiskit = rmsynth_circuit_to_qiskit(opt_circ)
    qasm_optimized = dumps(opt_qiskit)
    with open("../circuits/challenge_6_synth.qasm", "w") as f:
        f.write(qasm_optimized)
    print("Saved: challenge_6_synth.qasm")

if __name__ == "__main__":
    main()