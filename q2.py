#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 31 18:44:46 2026

@author: teehuien
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler.passes.synthesis.plugin import unitary_synthesis_plugin_names
from qiskit.qasm2 import dumps

theta = np.pi / 7
epsilon = 1e-6

qc = QuantumCircuit(2)
qc.cry(theta, 0, 1)

print("Installed synthesis plugins:", unitary_synthesis_plugin_names())

opt = transpile(
    qc,
    basis_gates=["h", "t", "tdg", "cx"],
    optimization_level=3,
    unitary_synthesis_method="gridsynth",                 # <-- Ross–Selinger
    unitary_synthesis_plugin_config={"epsilon": epsilon}  # supported by plugin
)

qasm_str = dumps(opt)
print(qasm_str)

print("T-count =", opt.count_ops().get("t", 0) + opt.count_ops().get("tdg", 0))

