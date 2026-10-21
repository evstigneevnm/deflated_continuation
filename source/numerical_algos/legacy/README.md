# Archived Linear Solvers

The original project solvers are retained for historical applications and explicit
old-versus-NMFD regression tests. New code uses `nmfd-linsolvers`.

- `include/numerical_algos/lin_solvers`: unchanged solver headers and preconditioners.
- `tests/lin_solvers`: original standalone tests and their fixtures.
- `../tests`: current NMFD integration, recovery, and migration-comparison tests.
- `../linear_solve_recovery.h`: active, solver-independent recovery support.

The archive has its own include root so historical `numerical_algos/lin_solvers/*`
includes remain valid without forwarding headers. The main Makefile enables this
root only for the explicit `LEGACY_LINEAR_SOLVER_TARGETS` allowlist, including the
frozen legacy Van der Pol example and deliberate comparison tests. Other historical
builds must explicitly add `-I source/numerical_algos/legacy/include`.

Archive files are excluded from automatic formatting. Their preservation does not
certify every historical application against current dependencies; only named CI
regressions are supported validation gates. Do not change numerical algorithms here.
