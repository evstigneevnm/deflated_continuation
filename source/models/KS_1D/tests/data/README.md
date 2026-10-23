# KS1D Linear-Solver Regressions

## Symmetric KS1D

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

## Full KS1D

`full_nmfd_continuation_reference.dat` records the parameter and normalized
reduced L2 norm of ten points generated with the pre-migration full-KS1D
binary built from parent commit `ec75b2c` (legacy BiCGStab(L)). It starts
from `stability_replay/curve03_state058.dat` at lambda=16.14704430336377.
The configuration is `KS1D_test_full.json` with N=128, knots [16, 16.2],
four steps per semicurve, initial step 0.02, maximum step 0.03, and no
deflation attempts. Existing symmetry policies and tolerances are unchanged.

The full replay compares this curve, checks its (1, 0) unstable signature,
verifies byte-identical stability restart, and exercises nonzero deflation.
The original state/transition fixtures and their manifest remain unchanged.

`test_KS1D_full_newton.cpp` uses the same seed with the production NMFD assembly
to check fixed and arclength corrections, frozen-chart/Jacobian lifetime,
Jacobian finite differences, orbit exclusion, failure rollback, and recovery.
Symmetry-aware storage uses gradient weight `N` for RMS distance paired with
the Euclidean scalar product. The workflow replay requires explicit NMFD Newton
selection and unchanged saved results on continuation/deflation restart; only
the enabled seed-attempt ledger may advance during deflation restart.
