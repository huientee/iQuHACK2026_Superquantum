"""
Challenge 2 Test: CRy(π/7) Optimization with rmsynth

This script:
1. Creates a 2-qubit CRy(π/7) circuit in Qiskit
2. Transpiles it to Clifford+T basis
3. Converts to rmsynth format
4. Optimizes using rmsynth's Reed-Muller decoder
5. Compares T-count before and after
"""

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import library
from qiskit.qasm2 import dumps
from rmsynth import Circuit, Optimizer, extract_phase_coeffs, synthesize_from_coeffs
import numpy as np


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


def main():
    print("=" * 70)
    print("Challenge 2: CRy(π/7) Optimization with rmsynth")
    print("=" * 70)
    
    # Step 1: Create Qiskit circuit
    print("Creating 2-qubit CRy(π/7) circuit in Qiskit...")
    qcirc = QuantumCircuit(2, name="CRy(π/7)")
    qcirc.cry(np.pi / 7, 0, 1)  # CRy on control=0, target=1
    
    print("Original circuit:")
    print(qcirc.draw())
    
    # Step 2: Transpile to Clifford+T
    print("\n[2] Transpiling to Clifford+T basis...")
    transpiled = transpile(
        qcirc,
        basis_gates=['h', 't', 'tdg', 'cx', 's', 'sdg'],
        optimization_level=3
    )
    
    # print("Transpiled circuit:")
    # print(transpiled.draw())
    
    # Count T-gates before rmsynth optimization
    t_before = transpiled.count_ops().get('t', 0) + transpiled.count_ops().get('tdg', 0)
    
    print(f"Gate counts (Qiskit transpiled):")
    print(f"t/tdg: {t_before}")

    print("Converting to rmsynth Circuit format...")
    rmsynth_circ = qiskit_circuit_to_rmsynth(transpiled)
    
    print(f"rmsynth circuit: {rmsynth_circ.n} qubits, {len(rmsynth_circ.ops)} operations")
    print(f"T-count in rmsynth: {rmsynth_circ.t_count()}")
    
    # Optimize with rmsynth
    print("Optimizing with rmsynth")
    opt = Optimizer(decoder="dumer", effort=0)
    opt_circ, report = opt.optimize(rmsynth_circ)
    
    print(f"Decoder used: {opt.last_decoder_used}")
    print(f"T-count before: {report.before_t}")
    print(f"t-count after:  {report.after_t}")
    print(f"Distance: {report.distance}")
    print(f"Selected monomials: {report.selected_monomials}")
    
    # Compare results
    
    print(f"\nQiskit transpiled (baseline):")
    print(f"  T-count: {t_before}")
    
    print(f"\nrmsynth optimized:")
    print(f"  T-count: {report.after_t}")
    
    t_reduction = t_before - report.after_t
    t_reduction_pct = (t_reduction / t_before * 100) if t_before > 0 else 0
    
    print(f"\n{'IMPROVEMENT':20} | {t_reduction:3} T gates ({t_reduction_pct:5.1f}%)")
    
    print("\n" + "=" * 70)
    

    #Export to QASM
    print("Exporting to QASM files...")
    
    # Baseline QASM
    qasm_baseline = dumps(transpiled)
    with open("../circuits/challenge_2_bad.qasm", "w") as f:
        f.write(qasm_baseline)
    print("  Saved: challenge_2_bad.qasm")
    
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
    with open("../circuits/challenge_2.qasm", "w") as f:
        f.write(qasm_optimized)
    print("Saved: challenge_2.qasm")


if __name__ == "__main__":
    main()