# Symmetric KS1D Linear-Solver Regression

`symmetric_lambda6.dat` is the unchanged state `KS1D_test_sym/1/1` from
`KS1D/N128/stability_20260824/symmetric/project.tar.xz`, published under
the `reference-data-20260824` release. The archived configuration uses
N=128, a=2, b=4; the first point of curve 1 has lambda=6.

The 63 sine coefficients exercise a nonzero equilibrium, its Newton
corrections, and ordinary, rank-one, and scalar-bordered Jacobian solves.
The complete archive remains under `data/reference`; CI needs only this state.

`symmetric_lambda6_curve.dat` contains the parameter and normalized reduced L2 norm
from the first seven rows of the same archive's `1/debug_curve.dat`.
The workflow replay compares both fields after nontrivial continuation,
then checks stability and byte-identical stability restart.
