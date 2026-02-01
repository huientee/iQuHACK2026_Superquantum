# Challenge solvers — IQuHACK 2026 Superquantum

**Single shared implementation.** All 11 challenges are implemented in **`utils.py`**; **`solve_all.py`** runs them and writes QASM. No per-challenge solver files.

## Files

- **`utils.py`** — Shared logic: cost counting, phase→Z8 conversion, rmsynth optimize, Qiskit helpers (Ch2,7,10), and **get_challenge_01_qasm() … get_challenge_11_qasm()** (GETTERS dict). Single source of truth for all challenges.
- **`solve_all.py`** — Runs GETTERS[1]…GETTERS[11], writes **`challenge_qasm/challenge_01.qasm` … `challenge_11.qasm`**, prints T/CNOT per challenge and total. Use **`--rmsynth-only`** to force Z8-approx for Ch2; use **`-o DIR`** for a different output directory.
- **`generate_and_compare.py`** — Generates two solution sets: **rmsynth-only** → `challenge_qasm_rmsynth/`, **Qiskit** → `challenge_qasm_qiskit/`, then prints a comparison table. Run from **challenge_solvers**: `python3 generate_and_compare.py`.
- **`test_rmsynth.py`** — Tests rmsynth (synthesize_from_coeffs, Optimizer) and utils integration (get_challenge_03_qasm).
- **`compare_qiskit_rmsynth.py`** — Compares our solution vs Qiskit for ch 2, 7, 10. Uses utils GETTERS and get_qiskit_ch*_qasm.

## Run (generate all QASM)

From **Superquantum** repo root:

```bash
python3 challenge_solvers/solve_all.py
```

Or from **challenge_solvers**:

```bash
python3 solve_all.py
```

- **Output:** `challenge_qasm/challenge_01.qasm` … `challenge_11.qasm` (gates: `h`, `t`, `tdg`, `cx` only).
- **Submit:** Upload the `.qasm` files at **iquhack.superquantum.io**.

**Default:** Best-efforts is **on** (rmsynth efforts 3, 4, 5 for diagonals; keep lowest T-count). Use `--no-best-efforts` to disable.

## Use only rmsynth (no Qiskit)

To generate solutions **without** using Qiskit (rmsynth + fixed circuits only):

```bash
python3 solve_all.py --rmsynth-only
```

- **Ch 1, 3, 4, 5, 6, 8, 9, 11:** Same as default (rmsynth or Clifford-only).
- **Ch 2:** Uses rmsynth on the CRz(π/7) diagonal (Z8 approximation) instead of Qiskit’s exact π/7 synthesis.
- **Ch 7, 10:** Use fixed Clifford+T circuits (no Qiskit state-prep or unitary synthesis).

Output is written to `challenge_qasm/` (or `-o DIR`). No Qiskit dependency required.

## Best implementation summary (per challenge PDF)

| Ch | Unitary | Method | rmsynth |
|----|---------|--------|---------|
| 1 | Controlled-Y | CY = (I⊗H)·CZ·(I⊗H); rmsynth on CZ diagonal; compare with S†·CX·S | ✓ |
| 2 | CRy(π/7) | Qiskit exact when available; else rmsynth on CRz(π/7) Z8 approx | ✓ fallback |
| 3 | exp(i π/7 Z⊗Z) | Qiskit exact when available; else rmsynth on diagonal | ✓ |
| 4 | exp(i π/7 (XX+YY)) | Same as Ch3 via (I⊗H)·exp(i π/7 ZZ)·(I⊗H); Qiskit or rmsynth | ✓ |
| 5 | exp(i π/4 (XX+YY+ZZ)) | = e^{iπ/4} SWAP; implement SWAP (3 CNOTs, 0 T) | Clifford only |
| 6 | exp(i π/7 (XX+ZI+IZ)) | Qiskit exact when available; else Trotter + rmsynth on ZZ block | ✓ |
| 7 | State prep (seed=42) | Qiskit initialize + decompose + transpile (best T over options) | — |
| 8 | 2-qubit QFT (ω=i) | H⊗H · controlled-phase(π/2) · SWAP; rmsynth on diagonal | ✓ |
| 9 | Structured unitary 2 | H·CX·(phase π/4 on \|11⟩)·CX·H; rmsynth on diagonal | ✓ |
| 10 | Random unitary (seed=42) | Qiskit random_unitary(4, seed=42) + decompose + transpile | — |
| 11 | 4-qubit diagonal φ(x) | Phases from PDF; rmsynth + T-depth schedule (min T and CNOT) | ✓ |

**Phase conventions (per challenge PDF):** Ch2 CRz(π/7): diagonal **[0, 0, −π/14, π/14]** (basis |00⟩,|01⟩,|10⟩,|11⟩). Ch3/4/6 exp(i π/7 Z⊗Z): diagonal **[π/7, −π/7, −π/7, π/7]** (Z⊗Z: +1 on |00⟩,|11⟩; −1 on |01⟩,|10⟩).

**Two solution sets (rmsynth approx vs Qiskit exact):** Generate both and compare T/CNOT:

```bash
python3 challenge_solvers/generate_and_compare.py
```

- Writes **rmsynth-only** (Z8 approx) to `challenge_qasm_rmsynth/` and **Qiskit** (exact π/7 for Ch2,3,4,6) to `challenge_qasm_qiskit/`, then prints a side-by-side comparison. With Qiskit installed, the Qiskit set has much higher T-count for Ch2,3,4,6.
- To generate only one set via solve_all: `--rmsynth-only -o challenge_qasm_rmsynth` or `-o challenge_qasm_qiskit` (default uses Qiskit when available).

## Test rmsynth

From **Superquantum** or **challenge_solvers**:

```bash
python3 challenge_solvers/test_rmsynth.py
```

- **Test 1:** `rmsynth.core.synthesize_from_coeffs`
- **Test 2:** `rmsynth.optimizer` (skipped if C++ rmcore not built)
- **Test 3:** utils integration (`get_challenge_03_qasm`)

## Dependencies

- **rmsynth:** `pip install -e rmsynth` (from Superquantum) for diagonal challenges 1, 2, 3, 4, 6, 8, 9, 11. Optional C++ rmcore for full optimization.
- **qiskit:** Optional; used for Ch2,3,4,6 (exact π/7 when available), Ch7 (state prep), Ch10 (random unitary). Fallbacks when Qiskit is missing.

## Documentation

- **CHALLENGE_EXPLANATION.md** — Full challenge description, rmsynth theory, how each challenge is solved.
- **EXPLANATION_EQUATIONS.md** — Short math summary (phase polynomials, Z8 vector).
