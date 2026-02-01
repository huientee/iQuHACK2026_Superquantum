OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
s q[1];
cx q[1],q[0];
s q[0];
cx q[1],q[0];