#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 23:30:56 2026

@author: teehuien
"""

#!/usr/bin/env python3
import numpy as np
from numpy.linalg import svd

# ============================================================
# 1) Target: Q10 random 2-qubit unitary (seed=42) from PDF
# ============================================================
U_target = np.array([
    [0.1448081895+0.1752383997j, -0.5189281551-0.5242425896j, -0.1495585824+0.3127549990j,  0.1691348143-0.5053863118j],
    [-0.9271743926-0.0878506193j, -0.1126033063-0.1818584963j, 0.1225587186+0.0964028611j, -0.2449850904-0.0504584131j],
    [-0.0079842758-0.2035507051j, -0.3893205530-0.0518092515j, 0.2605170566+0.3286402481j,  0.4451730754+0.6558933250j],
    [0.0313792249+0.1961395216j,  0.4980474972+0.0884604926j, 0.3407886532+0.7506609982j,  0.0146480652-0.1575584270j],
], dtype=complex)

# ============================================================
# 2) Gate matrices (single-qubit H, T, Tdg) and 2-qubit CNOT
# ============================================================
SQRT2_INV = 1/np.sqrt(2)
H = np.array([[1, 1],[1, -1]], dtype=complex) * SQRT2_INV
T = np.array([[1, 0],[0, np.exp(1j*np.pi/4)]], dtype=complex)
Tdg = np.conjugate(T).T
I2 = np.eye(2, dtype=complex)

# CNOT control q0 -> target q1 in computational basis |00>,|01>,|10>,|11>
CNOT = np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,0,1],
    [0,0,1,0],
], dtype=complex)

# ============================================================
# 3) Helpers: normalize global phase & operator norm distance
# ============================================================
def remove_global_phase(U):
    """Return U scaled so det becomes 1 (removes global phase up to sign)."""
    det = np.linalg.det(U)
    if det == 0:
        return U
    phase = det**(-0.5)
    return phase * U

def op_norm_distance_up_to_phase(U, V):
    """
    d = min_phi ||U - e^{i phi} V||_2
    We pick phi via Frobenius-optimal alignment: phi = arg(tr(U^† V)).
    Then compute operator norm (spectral norm) of the difference.
    """
    inner = np.trace(np.conjugate(U).T @ V)
    if abs(inner) < 1e-18:
        phi = 0.0
    else:
        phi = np.angle(inner)
    D = U - np.exp(1j*phi) * V
    # operator norm = largest singular value
    s = svd(D, compute_uv=False)
    return float(s[0])

def kron2(A, B):
    return np.kron(A, B)

# ============================================================
# 4) Build a library of 1-qubit Clifford+T circuits up to TMAX
#    using BFS with hashing modulo global phase.
# ============================================================
def unitary_key(U, tol=1e-10):
    """
    Hash key for a 2x2 unitary up to global phase:
    - remove global phase by det-normalization
    - round real/imag parts
    """
    U0 = remove_global_phase(U)
    v = np.concatenate([U0.real.flatten(), U0.imag.flatten()])
    v = np.round(v / tol).astype(np.int64)
    return tuple(v.tolist())

def gen_1q_library(TMAX=6, H_MAX=14):
    """
    Generate many distinct 1q unitaries with minimal T-count (up to TMAX).
    Gates allowed: H (0 T), T (1 T), Tdg (1 T).
    We cap H-only growth via H_MAX total length to keep the library manageable.
    """
    from collections import deque

    # state: (U, seq_str, tcount, length)
    start = (I2, [], 0, 0)
    q = deque([start])

    best = {}  # key -> (tcount, seq_list, U)
    best[unitary_key(I2)] = (0, [], I2)

    # Expand BFS; we allow revisiting if lower T-count found.
    while q:
        U, seq, tcount, length = q.popleft()

        # Stop conditions
        if tcount > TMAX or length > H_MAX:
            continue

        # Try appending each gate
        for gname, G, gT in (("h", H, 0), ("t", T, 1), ("tdg", Tdg, 1)):
            nt = tcount + gT
            nl = length + 1
            if nt > TMAX or nl > H_MAX:
                continue
            U2 = G @ U  # left-multiply (acts last in right-to-left convention for matrices)
            seq2 = seq + [gname]

            k = unitary_key(U2)
            prev = best.get(k)
            if (prev is None) or (nt < prev[0]) or (nt == prev[0] and len(seq2) < len(prev[1])):
                best[k] = (nt, seq2, U2)
                q.append((U2, seq2, nt, nl))

    # Convert to a list and sort by (tcount, length)
    lib = list(best.values())
    lib.sort(key=lambda x: (x[0], len(x[1])))
    return lib

# ============================================================
# 5) 2-qubit template: (A0⊗B0) CNOT (A1⊗B1) CNOT (A2⊗B2) CNOT (A3⊗B3)
#    All Ai,Bi are picked from the 1q library.
# ============================================================
TEMPLATE_BLOCKS = 4  # 4 local layers => 8 single-qubit choices total

def circuit_unitary(blocks):
    """
    blocks: list of length 4, each item is (A, B) 2x2 unitaries.
    Circuit: (A0⊗B0) CNOT (A1⊗B1) CNOT (A2⊗B2) CNOT (A3⊗B3)
    Returns 4x4 unitary.
    """
    U = kron2(blocks[0][0], blocks[0][1])
    U = CNOT @ U
    U = kron2(blocks[1][0], blocks[1][1]) @ U
    U = CNOT @ U
    U = kron2(blocks[2][0], blocks[2][1]) @ U
    U = CNOT @ U
    U = kron2(blocks[3][0], blocks[3][1]) @ U
    return U

def circuit_tcount(blocks_meta):
    """
    blocks_meta: list of length 4, each item is (tA, seqA, UA, tB, seqB, UB)
    """
    return sum(b[0] + b[3] for b in blocks_meta)

def blocks_to_qasm(blocks_meta):
    """
    Emit OpenQASM 2 using only h,t,tdg,cx.
    Note: our matrices were built by left-multiplying; to match that,
    we emit the sequence in the same order we stored (which is gate names in seq list).
    """
    lines = []
    lines.append("OPENQASM 2.0;")
    lines.append('include "qelib1.inc";')
    lines.append("qreg q[2];")

    def emit_1q(seq, qubit):
        for g in seq:
            lines.append(f"{g} q[{qubit}];")

    # Layer 0
    emit_1q(blocks_meta[0][1], 0)
    emit_1q(blocks_meta[0][4], 1)
    lines.append("cx q[0],q[1];")

    # Layer 1
    emit_1q(blocks_meta[1][1], 0)
    emit_1q(blocks_meta[1][4], 1)
    lines.append("cx q[0],q[1];")

    # Layer 2
    emit_1q(blocks_meta[2][1], 0)
    emit_1q(blocks_meta[2][4], 1)
    lines.append("cx q[0],q[1];")

    # Layer 3
    emit_1q(blocks_meta[3][1], 0)
    emit_1q(blocks_meta[3][4], 1)

    return "\n".join(lines) + "\n"

# ============================================================
# 6) Discrete coordinate-descent search
#    Objective: minimize (Tcount, distance)
# ============================================================
def search_best(Ut, lib, RESTARTS=25, ITERS=50, CANDIDATES_PER_UPDATE=400, seed=0):
    """
    lib entries: (tcount, seq, U)
    We keep a truncated candidate list for updates to stay fast.
    """
    rng = np.random.default_rng(seed)

    # Pre-split library by tcount for "T-minimizing" behavior
    # We'll mostly try low-T candidates first.
    lib_sorted = lib  # already sorted by (tcount, len)
    top_candidates = lib_sorted[:min(len(lib_sorted), CANDIDATES_PER_UPDATE)]

    best_pair = (10**9, 1e9)
    best_meta = None

    def eval_blocks_meta(meta):
        blocks = [(meta[i][2], meta[i][5]) for i in range(TEMPLATE_BLOCKS)]
        Uc = circuit_unitary(blocks)
        d = op_norm_distance_up_to_phase(Ut, Uc)
        tcnt = circuit_tcount(meta)
        return tcnt, d

    # identity entry should exist
    id_entry = next(x for x in lib_sorted if x[0] == 0 and len(x[1]) == 0)

    for r in range(RESTARTS):
        # random-ish init biased to low-T
        meta = []
        for _ in range(TEMPLATE_BLOCKS):
            a = top_candidates[rng.integers(0, len(top_candidates)//3 + 1)]
            b = top_candidates[rng.integers(0, len(top_candidates)//3 + 1)]
            meta.append((a[0], a[1], a[2], b[0], b[1], b[2]))

        # occasional pure identity start
        if r % max(1, RESTARTS//5) == 0:
            meta = []
            for _ in range(TEMPLATE_BLOCKS):
                a = id_entry
                b = id_entry
                meta.append((a[0], a[1], a[2], b[0], b[1], b[2]))

        cur_t, cur_d = eval_blocks_meta(meta)

        for it in range(ITERS):
            improved = False

            # update each of 8 positions (A/B in each of 4 layers)
            for layer in range(TEMPLATE_BLOCKS):
                for which in (0, 1):  # 0 = A on q0, 1 = B on q1
                    best_local = (cur_t, cur_d)
                    best_choice = None

                    # try candidates in increasing Tcount order
                    for cand in top_candidates:
                        new_meta = list(meta)
                        if which == 0:
                            # replace A
                            new_meta[layer] = (cand[0], cand[1], cand[2],
                                               new_meta[layer][3], new_meta[layer][4], new_meta[layer][5])
                        else:
                            # replace B
                            new_meta[layer] = (new_meta[layer][0], new_meta[layer][1], new_meta[layer][2],
                                               cand[0], cand[1], cand[2])

                        tcnt, dist = eval_blocks_meta(new_meta)

                        # Lexicographic: Tcount first, then distance
                        if (tcnt < best_local[0]) or (tcnt == best_local[0] and dist < best_local[1]):
                            best_local = (tcnt, dist)
                            best_choice = new_meta

                    if best_choice is not None and best_local <= (cur_t, cur_d):
                        # accept improvement (or equal T but smaller dist)
                        if best_local != (cur_t, cur_d):
                            improved = True
                        meta = best_choice
                        cur_t, cur_d = best_local

            if not improved:
                break

        if (cur_t < best_pair[0]) or (cur_t == best_pair[0] and cur_d < best_pair[1]):
            best_pair = (cur_t, cur_d)
            best_meta = meta
            print(f"[restart {r+1}/{RESTARTS}] new best: T={best_pair[0]}, opdist={best_pair[1]:.6e}")

    return best_pair, best_meta

# ============================================================
# 7) Main: build library, search, emit qasm
# ============================================================
if __name__ == "__main__":
    # --- Tune these ---
    TMAX_1Q = 7            # higher => better accuracy, more T
    RESTARTS = 30          # more => better chance, slower
    ITERS = 60             # more => better refinement, slower
    CANDS = 600            # higher => better, slower

    print("Generating 1-qubit Clifford+T library...")
    lib = gen_1q_library(TMAX=TMAX_1Q, H_MAX=18)
    print(f"Library size: {len(lib)} (TMAX_1Q={TMAX_1Q})")

    print("Searching 2-qubit circuit in 3-CNOT template...")
    (bestT, bestD), best_meta = search_best(
        U_target, lib,
        RESTARTS=RESTARTS,
        ITERS=ITERS,
        CANDIDATES_PER_UPDATE=CANDS,
        seed=42
    )

    if best_meta is None:
        raise RuntimeError("Search failed unexpectedly.")

    qasm = blocks_to_qasm(best_meta)

    print("\n==== BEST FOUND ====")
    print(f"T-count: {bestT}")
    print(f"Operator-norm distance (phase-optimized): {bestD:.12e}")
    print("\n==== OPENQASM 2 ====")
    print(qasm)
