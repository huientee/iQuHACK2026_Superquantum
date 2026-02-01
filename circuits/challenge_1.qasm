OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
sdg q[1];
s q[0];
cx q[0],q[1];
s q[1];