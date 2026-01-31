# Challenge 1, 2 & 3 solvers

Self-contained solvers for IQuHACK Superquantum challenges 1, 2 and 3, using **rmsynth** for Clifford+T synthesis.

## Files

- **`solve_01.py`** — Challenge 1: Controlled-Y (CY). Decomposition: H on target, CZ diagonal, H on target; rmsynth optimizes the diagonal.
- **`solve_02.py`** — Challenge 2: Controlled-Ry(π/7). Same structure as CY; uses CRz(π/7) diagonal [0,0,−π/14,π/14]. With Qiskit: exact Clifford+T; without: rmsynth on Z8-rounded phases.
- **`solve_03.py`** — Challenge 3: exp(i π/7 Z⊗Z). Diagonal phases [π/7, −π/7, −π/7, π/7]. With Qiskit: exact synthesis; without: rmsynth on Z8.

## Run

From the **repo root** (parent of `challenge_solvers`):

```bash
pip install -e rmsynth   # if not already installed
python3 challenge_solvers/solve_01.py   # writes challenge_qasm/challenge_01.qasm
python3 challenge_solvers/solve_02.py   # writes challenge_qasm/challenge_02.qasm
python3 challenge_solvers/solve_03.py   # writes challenge_qasm/challenge_03.qasm
```

Or from **challenge_solvers**:

```bash
python3 solve_01.py
python3 solve_02.py
python3 solve_03.py
```

Output: `challenge_qasm/challenge_01.qasm`, `challenge_02.qasm`, `challenge_03.qasm` (gates: `h`, `t`, `tdg`, `cx` only).

## Dependencies

- **rmsynth:** `pip install -e rmsynth` from the repo root. Optional C++ rmcore for full optimization.
- **qiskit:** Optional; used by `solve_02.py` and `solve_03.py` for exact π/7 synthesis when available.
