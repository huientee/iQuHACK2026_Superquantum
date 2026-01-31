"""Reference unitary matrices for the challenge tasks."""

# I generated this with Claude; so double-check at some pt.

import numpy as np

# Task 1: Controlled-Y Gate
U_controlled_y = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, -1j],
    [0, 0, 1j, 0]
], dtype=complex)

# Task 2: Controlled-Ry(π/7)
U_controlled_ry = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, -0.22 - 0.97j, 0],
    [0, 0, 0, 0.22 + 0.97j]
], dtype=complex)

# Task 7: State preparation (from Qiskit random_statevector(4, seed=42))
# Maps |00⟩ to superposition
STATE_PREP_COEFFS = [
    0.1061479384 - 0.679641467j,
    -0.3622775887 - 0.453613136j,
    0.2614190429 + 0.0445330969j,
    0.3276449279 - 0.1101628411j
]

# Task 8: Structured unitary 1 (looks like a Fourier-like matrix)
U_structured_1 = np.array([
    [1, 1, 1, 1],
    [1, 1j, -1, -1j],
    [1, -1, 1, -1],
    [1, -1j, -1, 1j]
], dtype=complex) / 2

# Task 10: Random unitary (from Qiskit random_unitary(4, seed=42))
U_random = np.array([
    [0.1448081895 + 0.1752383997j, -0.5189281551 - 0.5242425896j, -0.1495585824 + 0.312754999j, 0.1691348143 - 0.5053863118j],
    [-0.9271743926 - 0.0878506193j, -0.1126033063 - 0.1818584963j, 0.1225587186 + 0.0964028611j, -0.2449850904 - 0.0504584131j],
    [-0.0079842758 - 0.2035507051j, -0.3893205530 - 0.0518092515j, 0.2605170566 + 0.3286402481j, 0.4451730754 + 0.6558933250j],
    [0.0313792249 + 0.1961395216j, 0.4980474972 + 0.0884604926j, 0.3407886532 + 0.7506609982j, 0.0146480652 - 0.1575584270j]
], dtype=complex)

# Task 11: 4-qubit diagonal unitary phases
DIAGONAL_PHASES = {
    0b0000: 0,
    0b0001: np.pi,
    0b0010: 5/4 * np.pi,
    0b0011: 7/4 * np.pi,
    0b0100: 5/4 * np.pi,
    0b0101: 7/4 * np.pi,
    0b0110: 3/2 * np.pi,
    0b0111: 3/2 * np.pi,
    0b1000: 5/4 * np.pi,
    0b1001: 7/4 * np.pi,
    0b1010: 3/2 * np.pi,
    0b1011: 3/2 * np.pi,
    0b1100: 3/2 * np.pi,
    0b1101: 3/2 * np.pi,
    0b1110: 7/4 * np.pi,
    0b1111: 5/4 * np.pi,
}
