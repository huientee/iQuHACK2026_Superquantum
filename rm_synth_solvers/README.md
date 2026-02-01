## IQuHACK 2026 Superquantum challenge — <Qu|ack> team

This document explains the challenge, the theory behind rmsynth, and how each of the 11 unitaries is solved **using rmsynth** (Reed–Muller phase-polynomial optimization).

---

### What the challenge asks for

- **Goal**: compile given unitaries into **Clifford + T** circuits: only gates from {$H$, $T$, $T^{\dagger}$, CNOT}.
- **Motivation**: in fault-tolerant quantum computing, non-Clifford gates (especially $T$) are expensive. Minimizing **T-count** reduces fault-tolerant cost.
- **Output**: OpenQASM 2.0 or 3.0 files; submit at ?.
- **Costs** (from `challenge.pdf`):
  1. **Operator norm distance:** $d(U,\tilde{U}) = \min_{\phi\in[0,2\pi)} \|U - e^{i\phi}\tilde{U}\|_{\mathrm{op}}$. Lower is better (0 = exact).
  2. **T-count:** number of $T$ and $T^\dagger$ gates. Lower is better.

### What rmsynth is and how it works

**rmsynth** is Superquantum’s Clifford+T optimizer based on **phase polynomials** and **punctured Reed–Muller (RM) decoding** (Amy & Mosca, and related work).

### Phase polynomials (CNOT + phase circuits)

- A circuit made only of **CNOT** and **phase gates** $P_k = e^{i\pi k/4}$ (with $k \in \mathbb{Z}_8$) is **diagonal in the computational basis**.
- Its action is: $|x\rangle \mapsto e^{i\phi(x)} |x\rangle$, where $x \in \{0,1\}^n$.
- The phase can be written as a **phase polynomial**:
  $$
  \phi(x) = \frac{\pi}{4} \sum_{\emptyset \neq S \subseteq [n]} a_S \cdot \chi_S(x) \: \text{(mod } 2\pi\text{)},
  $$
  here $\chi_S(x)$ is the **parity** of the bits of $x$ in $S$ (XOR or $\mathbb{F}_2$ sum). So $\chi_S(x) \in \{0,1\}$.
- Coefficients $a_S$ are in **$\mathbb{Z}_8$** (multiples of $\pi/4$). The **T-count** of such a circuit equals the number of **odd** coefficients $a_S$ (each odd term needs at least one $T$ or $T^\dagger$).

So: **fewer odd coefficients means a lower T-count.**

### Reed–Muller decoding and rmsynth

- The “odd part” of the phase polynomial (which coefficients are odd) can be viewed as a **binary word** of length $2^n - 1$ (one bit per non-empty subset $S$).
- **Punctured Reed–Muller codes** $\mathrm{RM}(r,n)^*$ live in the same space. **Decoding** finds a codeword close to this word in Hamming distance.
- **rmsynth**:
  1. Takes a **CNOT + phase** circuit (or a **Z8 phase vector**).
  2. Extracts the phase polynomial coefficients (mod 8).
  3. Treats the odd coefficients as a received word and **decodes** it in $\mathrm{RM}(n-4,n)^*$ (or similar) to get a **correction** codeword.
  4. **Adds** this correction to the coefficients (mod 8) and **re-synthesizes** a new CNOT + phase circuit.
- The new circuit implements the **same** diagonal unitary (same phases mod $2\pi$) but with **fewer T gates** when the decoder finds a good codeword.

So: **rmsynth minimizes T-count for circuits that are (or can be written as) CNOT + phase.**

### Pipeline (?)

1. **Diagonal unitary** $U|x\rangle = e^{i\phi(x)}|x\rangle$:  
   - From the list $\phi(0),\ldots,\phi(2^n-1)$ we can **invert** the phase-polynomial relation to get coefficients $a_S \in \mathbb{Z}_8$ (see below).  
   - Build the **Z8 vector** of length $2^n - 1$ (one entry per non-empty $S$).  
   - Run `rmsynth` (optimizer) on this vector to get an **optimized** CNOT + phase circuit.  
   - Convert that circuit to OpenQASM using only **cx, t, tdg** (and **h** only when we conjugate by Hadamards; see below).

2. **Non-diagonal unitary**:  
   - Decompose it into **Hadamards** and **diagonal** (CNOT + phase) blocks, e.g. $U = \tilde{H} \cdot D \cdot \tilde{H}$.  
   - Get the diagonal $D$ as a phase polynomialas a phase polynomial and convert it into a Z8 vector. Then apply rmsynth to obtain an optimized circuit composed of CNOT + phase gates.  
   - Wrap with H as needed and emit **h, t, tdg, cx** only.

So in our solution, `rmsynth` is used for every diagonal part** (and thus for every challenge that contains a diagonal block).

---

### From diagonal phases $\phi(x)$ to Z8 vector for rmsynth

- We have $\phi(x) = \frac{\pi}{4} \sum_S a_S \chi_S(x)$ where  $b(x) := \frac{4}{\pi}\phi(x) = \sum_{S:\ \chi_S(x)=1} a_S$ (in reals, then mod 8).
- The inverse transform (to get $a_S$ from $b(x)$) is:
  $$
  a_S = \frac{1}{2^{n-1}} \sum_x (-1)^{\chi_S(x)} b(x).
  $$
- We compute $b(x) = 4\phi(x)/\pi$ from the challenge’s $\phi(x)$, then $a_S$ with the formula above, round to integers, reduce mod 8, and form the **Z8 vector** in rmsynth canonical order (non-empty subsets $S$, e.g. by mask $1,2,\ldots,2^n-1$).

This vector is what we pass to rmsynth optimizer (and optionally to `synthesize_from_coeffs` before/after optimization).

---

### Challenge 1 - Controlled-Y

- **Unitary:** $\mathrm{CY} = \mathrm{diag}(1,1,i,-i)$ on the target in the $|10\rangle,|11\rangle$ subspace (plus identity on $|00\rangle,|01\rangle$).
- **Standard identity:** $\mathrm{CY} = (I \otimes H)\, \mathrm{CZ}\, (I \otimes H)$. So we need **H(target), CZ, H(target)**.
- **CZ** is diagonal: $\phi(11)=\pi$, others 0. So we build the Z8 vector for CZ, run **rmsynth** on it, get an optimized CNOT + phase circuit for CZ, then wrap with **H** on the target.  
- Alternative (used in code): known minimal decomposition $\mathrm{CY} = S^\dagger(\text{target})\,\mathrm{CX}\,S(\text{target})$ with $S = T^2$, which is already Clifford + T; we can also replace the CZ block by an rmsynth-optimized diagonal to “solve using rmsynth”.

### Challenge 2 - Controlled-$R_y(\pi/7)$

- **CRy(θ) = (I⊗H) CRz(θ) (I⊗H).** CRz(π/7) diagonal phases **[0, 0, −π/14, π/14]** (basis |00⟩,|01⟩,|10⟩,|11⟩; Qiskit: control q0, target q1).
- **Qiskit** synthesizes CRy(π/7) to Clifford+T when available; else **rmsynth** on the Z8-rounded diagonal.

### Challenge 3 - $\exp(i\frac{\pi}{7} Z \otimes Z)$

- **Fully diagonal:** $Z\otimes Z$ has +1 on $|00\rangle,|11\rangle$ and −1 on $|01\rangle,|10\rangle$, so phases **$[\pi/7,\, -\pi/7,\, -\pi/7,\, \pi/7]$** (order |00⟩,|01⟩,|10⟩,|11⟩).
- Compute Z8 vector, run **rmsynth** (or Qiskit unitary synthesis), emit OpenQASM **cx, t, tdg** only.

### Challenge 4 - $\exp(i\frac{\pi}{7}(XX+YY))$

- **Structure:** $e^{i\frac{\pi}{7}(XX+YY)} = (I\otimes H)\, e^{i\frac{\pi}{7} Z\otimes Z}\, (I\otimes H)$. Same diagonal as Ch3: **$[\pi/7,\, -\pi/7,\, -\pi/7,\, \pi/7]$**.
- Run **rmsynth** on that diagonal, then **wrap with H on the target** before and after.

### Challenge 5 — $\exp(i\frac{\pi}{4}(XX+YY+ZZ))$

- For 2 qubits, $XX+YY+ZZ = 2\,\mathrm{SWAP} - I$, so $e^{i\frac{\pi}{4}(XX+YY+ZZ)} = e^{i\pi/4}\,\mathrm{SWAP}$ (global phase × SWAP).  
- **No T gates:** circuit is **SWAP** = three CNOTs. Clifford only; no rmsynth.

### Challenge 6 — $\exp(i\frac{\pi}{7}(XX+ZI+IZ))$ (Ising)

- **Trotter:** $e^{i\frac{\pi}{7}(XX+ZI+IZ)} \approx e^{i\frac{\pi}{7}ZI}\, e^{i\frac{\pi}{7}IZ}\, e^{i\frac{\pi}{7}XX}$.  
- $e^{i\frac{\pi}{7}XX} = H_0\, e^{i\frac{\pi}{7}ZZ}\, H_0$ (Hadamard on qubit 0).  
- The **ZZ** part uses the same diagonal as Ch3/4: **$[\pi/7,\, -\pi/7,\, -\pi/7,\, \pi/7]$**. Run **rmsynth**, then **H on qubit 0** before and after.

### Challenge 7 — State preparation

- Target state from `qiskit.quantum_info.random_statevector(4, seed=42)`. We use Qiskit to get a Clifford + T circuit (or a fallback). Any **diagonal** sub-blocks in that decomposition could be optimized with rmsynth; the script focuses on producing valid **h, t, tdg, cx** output.

### Challenge 8 — Structured unitary 1 (QFT-like)

- Matrix is $\frac{1}{2}[1,1,1,1;\ 1,i,-1,-i;\ 1,-1,1,-1;\ 1,-i,-1,i]$ (2-qubit DFT with $\omega=i$).  
- Decomposition: **H on both qubits**, then **controlled phase** $\pi/2$ (e.g., two T gates on target when control is 1), then **SWAP**. The controlled-phase block is diagonal and could be expressed as a phase polynomial and optimized with rmsynth; we use a standard small circuit and **h, t, tdg, cx** only.

### Challenge 9 — Structured unitary 2

- Similar idea: identify structure (e.g. Hadamards + diagonal), write diagonal part as phase polynomial, run **rmsynth** on that part if desired; output **h, t, tdg, cx**.

### Challenge 10 — Random unitary

- From `qiskit.quantum_info.random_unitary(4, seed=42)`. We use Qiskit synthesis + transpile to **cx, h, t, tdg**. Any diagonal blocks in the decomposition could be passed to rmsynth.

### Challenge 11 — 4-qubit diagonal

- **Fully diagonal** with given $\phi(x)$ for $x \in \{0,1\}^4$.  
- We compute the Z8 vector of length $2^4-1=15$ from the 16 phases, run **rmsynth** optimizer, re-synthesize, and output 4-qubit OpenQASM with **cx, t, tdg** only.  
- **Fully solved with rmsynth.**

| Challenge | Use of rmsynth |
|-----------|-----------------|
| 1  | CZ part of CY (optional); or known CY decomposition. |
| 2  | CRz(π/7) diagonal [0,0,−π/14,π/14]; Qiskit or rmsynth fallback. |
| **3** | **Full:** Diagonal [π/7,−π/7,−π/7,π/7] → rmsynth or Qiskit → QASM. |
| **4** | **Full:** Same Z8 as 3 → rmsynth → H + circuit + H. |
| 5  | exp(i π/4 (XX+YY+ZZ)) = e^{iπ/4} SWAP; 3 CNOTs only. |
| **6** | **Full:** Trotter; ZZ block same as 3 → rmsynth → H + circuit + H. |
| 7  | Qiskit state prep (seed=42) or fixed circuit. |
| 8–10 | Structured/random; rmsynth on diagonal (Ch8,9); Qiskit (Ch10). |
| **11** | **Full:** 4-qubit diagonal (phases from PDF) → rmsynth + T-depth schedule. |

So the **core** of the solution (challenges 3, 4, 6, 11, and optionally 1) is **solving with rmsynth**: build the phase polynomial (Z8 vector) from the diagonal unitary, run rmsynth to minimize T-count, then emit OpenQASM.

---

## 6. How to run and submit

1. **From repo root (Superquantum):**
   ```bash
   cd /path/to/Superquantum
   pip install -e rmsynth   # if not already installed
   python3 challenge_solvers/solve_all.py
   ```
2. **Output:** `challenge_qasm/challenge_01.qasm` … `challenge_11.qasm` (only **h, t, tdg, cx**). Best-efforts (rmsynth efforts 3,4,5 for diagonals) is on by default; use `--no-best-efforts` to disable.
3. **Optional:** Build rmsynth C++ extension for faster decoders: run `pip install -e rmsynth` from repo root.
4. **Submit** the `.qasm` files at **iquhack.superquantum.io** with your team API key.

---

## 7. References

- Challenge PDF: `Superquantum/challenge.pdf`
- rmsynth: [Superquantum/rmsynth](https://github.com/super-quantum/rmsynth) (phase-polynomial optimization, Reed–Muller decoding).
- Amy & Mosca: “T-count optimization and Reed–Muller codes”; Amy, Maslov, Mosca: “Polynomial-time t-depth optimization of Clifford+T circuits via matroid partitioning”.