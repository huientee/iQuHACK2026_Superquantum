import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

try:
    from qiskit.synthesis import TwoQubitBasisDecomposer
except Exception:
    from qiskit.synthesis.two_qubit import TwoQubitBasisDecomposer


U = np.array([
    [ 0.1448081895 + 0.1752383997j, -0.5189281551 - 0.5242425896j, -0.1495585824 + 0.3127549990j,  0.1691348143 - 0.5053863118j],
    [-0.9271743926 - 0.0878506193j, -0.1126033063 - 0.1818584963j,  0.1225587186 + 0.0964028611j, -0.2449850904 - 0.0504584131j],
    [-0.0079842758 - 0.2035507051j, -0.3893205530 - 0.0518092515j,  0.2605170566 + 0.3286402481j,  0.4451730754 + 0.6558933250j],
    [ 0.0313792249 + 0.1961395216j,  0.4980474972 + 0.0884604926j,  0.3407886532 + 0.7506609982j,  0.0146480652 - 0.1575584270j],
], dtype=complex)


def op_norm_distance_mod_phase(Ua, Va):
    """Operator (spectral) norm distance for 4x4 matrices, mod global phase."""
    X = Va.conj().T @ Ua
    phi = np.angle(np.trace(X))
    Vp = np.exp(1j * phi) * Va
    svals = np.linalg.svd(Ua - Vp, compute_uv=False)
    return float(np.max(svals))

def op_norm_distance_mod_phase_2x2(Ua, Va):
    """Same but for 2x2; used for 1-qubit gate matching."""
    X = Va.conj().T @ Ua
    phi = np.angle(np.trace(X))
    Vp = np.exp(1j * phi) * Va
    svals = np.linalg.svd(Ua - Vp, compute_uv=False)
    return float(np.max(svals))


def build_1q_library(m=3):
    # Precompute T^k unitaries
    T = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=complex)
    H = (1/np.sqrt(2)) * np.array([[1, 1],[1, -1]], dtype=complex)
    Tk = [np.linalg.matrix_power(T, k) for k in range(8)]

    lib = []  # list of (Umat, circuit, t_count)

    def rec(exps):
        if len(exps) == m + 1:
            # Build unitary U = T^a0 * Î _{i=1..m} (H T^ai)
            Umat = Tk[exps[0]]
            qc = QuantumCircuit(1)
            # Apply T^a0 (as t/tdg sequences)
            Umat = Tk[exps[0]]
            # Build circuit:
            for _ in range(exps[0] % 8):
                qc.t(0)
            qc = None  # placeholder; we rebuild cleanly below

            # Compose unitary
            Umat = Tk[exps[0]]
            for i in range(1, m + 1):
                Umat = Umat @ H @ Tk[exps[i]]

            # Count Ts minimally (k -> min(k, 8-k) using tdg)
            t_count = sum(min(k % 8, (8 - (k % 8)) % 8) for k in exps)

            lib.append((Umat, exps, t_count))
            return
        for a in range(8):
            rec(exps + [a])

    rec([])

    # Helper to rebuild a circuit from exponents with minimal t/tdg usage
    def emit_exponent(qc, k):
        k = k % 8
        if k <= 4:
            for _ in range(k):
                qc.t(0)
        else:
            for _ in range(8 - k):
                qc.tdg(0)

    # Convert stored exponent tuples into actual circuits (with minimal t/tdg)
    lib2 = []
    for Umat, exps, t_count in lib:
        qc = QuantumCircuit(1)
        emit_exponent(qc, exps[0])
        for i in range(1, m + 1):
            qc.h(0)
            emit_exponent(qc, exps[i])
        lib2.append((Umat, qc, t_count))

    return lib2


def best_1q_approx(U_target_2x2, lib):
    best = None
    best_d = 1e9
    for Ucand, qccand, tcnt in lib:
        d = op_norm_distance_mod_phase_2x2(U_target_2x2, Ucand)
        if d < best_d:
            best_d = d
            best = (qccand, tcnt, best_d)
    return best  # (qc1, t_count, dist)


def solve_q10(U4, lib_m=3):
    cx_basis = QuantumCircuit(2)
    cx_basis.cx(0, 1)
    decomp2 = TwoQubitBasisDecomposer(Operator(cx_basis))
    qc_exact = decomp2(Operator(U4))

    lib = build_1q_library(m=lib_m)

    qc_out = QuantumCircuit(2)

    for inst, qargs, _ in qc_exact.data:
        name = inst.name.lower()
        if name in ("cx", "cnot"):
            q0 = qc_exact.find_bit(qargs[0]).index
            q1 = qc_exact.find_bit(qargs[1]).index
            qc_out.cx(q0, q1)
        elif len(qargs) == 1:
            q = qc_exact.find_bit(qargs[0]).index

            tmp = QuantumCircuit(1)
            tmp.append(inst, [0])
            U1 = Operator(tmp).data

            approx_qc1, _, _ = best_1q_approx(U1, lib)

            qc_out.append(approx_qc1.to_instruction(), [q])
        else:
            raise RuntimeError(f"Unexpected multi-qubit gate: {inst.name}")

    allowed = {"h", "t", "tdg", "cx"}
    ops = set(qc_out.count_ops().keys())
    if not ops.issubset(allowed):
        raise RuntimeError(f"Disallowed gates produced: {ops - allowed}")

    U_tilde = Operator(qc_out).data
    dist = op_norm_distance_mod_phase(U4, U_tilde)
    counts = qc_out.count_ops()
    t_count = int(counts.get("t", 0) + counts.get("tdg", 0))

    return qc_out, dist, t_count, counts


if __name__ == "__main__":
    qc_out, dist, t_count, counts = solve_q10(U, lib_m=3)

    print("=== Results (Regime A) ===")
    print("Operator-norm distance (mod global phase):", dist)
    print("T-count:", t_count)
    print("Gate counts:", counts)

    print("\n=== OpenQASM 2.0 ===")
    print(qc_out.qasm())