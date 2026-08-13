# Stability Analysis

The current bifurcation-diagram stability path is matrix free and
hardware agnostic. The nonlinear operator exposes only its ordinary real
linearization. Spectral transformations, complex arithmetic, iterative
factor solves, eigensolver ownership, scan aggregation, and stability
classification belong to the stability layer.

## Modern Pipeline

The main layers are:

1. `analysis/stability_evaluator.h` sets the linearization point, creates
   an admissible initial vector, executes an eigensolver adapter, and
   classifies the recovered physical spectrum.
2. `analysis/matrix_free_stability_scan.h` runs one or more transformed
   Krylov-Schur solves for each configured spectral shift. Multiple
   probes recover repeated eigenspaces.
3. `analysis/eigenvector_rank_aggregator.h` accepts only numerically
   independent recovered Ritz vectors within each eigenvalue cluster.
4. `analysis/spectrum_scan_aggregator.h` merges equivalent estimates
   across shifts without deleting geometric multiplicity.
5. `stability_analysis.hpp` is the bifurcation-diagram facade. It also
   owns transition refinement through the ordinary Newton solver.
6. `main/stability_continuation.hpp` traverses saved branches and writes
   transactional stability curves.

The transformation implementations live under
`eigensolvers/transformations`. Complexification is internal to that
layer: a complex matrix-free action is evaluated by applying the real
linearization separately to the real and imaginary components. Complex
types are not part of the nonlinear-operator interface.

## Model Assembly

A model-specific stability executable supplies:

- vector operations and host small-dense operations;
- the real nonlinear operator;
- a linearization provider;
- an affine inverse provider used by transformed inner solves;
- a probe generator that respects model constraints;
- `matrix_free_stability_config` parsed from JSON.

For an unconstrained problem the nonlinear operator can also be the
linearization provider. A quotient problem supplies projected providers
from `source/symmetry/linearization`. Physical sign conventions are
expressed with `scaled_real_operator` and
`scaled_real_affine_inverse_provider`; they are not encoded in the
nonlinear operator.

`source/models/KS_1D/KS1D_stability.cpp` is the ordinary example.
`source/models/KS_1D/KS1D_full_stability.cpp` is the projected
continuous-symmetry example.

Low-dimensional models may use the same traversal without a Krylov
method. `direct_scalar_eigensolver` is used by the circle and
star-shaped drivers, while still exercising generic classification,
transition refinement, persistence, and restart. Bratu uses the
matrix-free shifted pipeline in production and
`host_dense_operator_eigensolver` as a LAPACK validation oracle.

## Configuration Contract

The modern JSON configuration controls:

- transition-refinement maximum iterations and parameter tolerance;
- physical linearization scale;
- transformation type and spectral shifts;
- outer Krylov-Schur dimensions and tolerances;
- inner iterative factor solves;
- required successful scans and recovered eigenpairs;
- probe count and probe-failure policy;
- eigenvalue clustering and eigenvector-rank tolerances.
- optional residual-validated Ritz-subspace recycling between nearby
  continuation states.

`require_all_scans` and `require_all_probes` establish computational
coverage of the configured work. They do not prove that the selected
shifts enclose every unstable eigenvalue. Shift selection must still be
validated for each model and parameter range.

The probe count must be large enough to expose expected geometric
multiplicity. `minimum_successful_probes` permits a bounded number of
failed probe solves without accepting fewer probes than the configured
coverage floor. The stability facade applies the nonlinear operator's
stability-specific randomizer to every generated probe when the
eigensolver supports probe injection.

For a finite equivariance group, unrelated random probes are not always
the preferred way to recover symmetry-protected multiplicity. A model
may preconfigure the scan's indexed probe generator with selected group
actions. The initial vector is probe zero and subsequent probes receive
both their index and that initial vector. KS2D uses independent random
seeds paired with their axis-swapped copies. Its half-shift actions only
change Fourier signs and therefore do not add eigenspace rank.

Dimension changes are confirmed before transition refinement. A
`dimension_guarded_eigensolver` uses its exact host-dense solver for
this confirmation when the state dimension is below the configured
small-system limit, while leaving ordinary branch traversal on the
matrix-free path. When no dedicated confirmation solver is available,
`transition_classification_confirmations` independent matrix-free
classifications must reach consensus. If the first runs disagree, the
configured classification retries are used as additional confirmation
runs. Only the repeatedly recovered signature with the largest real
unstable-subspace dimension is accepted; an unconfirmed larger
signature or conflicting signatures of equal dimension abort the curve
transaction. This reflects the one-sided failure mode of a
residual-validated Krylov solve: it may miss a direction in a repeated
eigenspace, but one run is not allowed to introduce an extra direction.
Confirmed endpoint signatures also replace provisional regular-point
classifications in the pending stability curve.

Confirmed transition refinement uses a single classification at an
ordinary midpoint. Full consensus is requested only for an incomplete
classification, a signature outside both endpoint dimensions, or the
final accepted midpoint. This preserves the multiplicity guard without
doubling every matrix-free solve in a long refinement.

`inner_solver.basis_retry_sizes` is an optional strictly increasing
list of GMRES restart dimensions. When a healthy affine factor fails,
the scan retries that factor with each larger basis before trying the
configured shift perturbations. Each retry owns a fresh solver
assembly, so the larger Krylov storage exists only while it is needed.
Physical residual tolerances are not relaxed.

## Ritz-Subspace Recycling

`matrix_free_eigensolver.recycling` reuses converged physical Ritz
vectors while traversing a continuous curve. It does not reuse an old
Arnoldi factorization after the Jacobian changes. Before each solve,
the cached real invariant vectors are applied to the current real
linearization, their current complex Rayleigh quotients are recomputed,
and they are retained only when the resulting physical eigenpair
residuals satisfy the configured absolute or relative tolerance.
Accepted vectors are combined with a nonzero fresh random component
controlled by `innovation_weight`.

Cache updates follow the same transaction as stability classification.
Independent confirmation attempts see the last committed cache;
vectors recovered by a failed or disputed classification are rolled
back. The cache is reset at the start of every curve and at saved curve
breaks, so vectors are never carried across unrelated semicurves.

The policy is disabled by default. Its fields are:

- `enabled`;
- `maximum_vectors`;
- `innovation_weight` in `(0, 1]`;
- `absolute_residual_tolerance`;
- `relative_residual_tolerance`.

The implementation uses only the vector-space interface and ordinary
real operator actions. The same path therefore supports serial, OMP,
CUDA, and other SCFD-backed vector spaces.

## Persistence

The stability archive stores completed curves only. A curve is assembled
transactionally: failed curves are abandoned and pending bifurcation
state files are removed. Restart resumes at the first unprocessed
bifurcation-diagram curve and is a no-op when coverage is complete.

The serialized archive, each derived `debug_curve_stability.dat` file,
and each `debug_curve_stability_plot.dat` sidecar are written through
sibling temporary files and committed only after successful writes.
The plotting sidecar is deliberately not serialized. It records the
source bifurcation-diagram index, dimensions on both sides of a
transition, the broad transition class, and the norm vector evaluated
at the refined transition state. This lets visualization place an event
at its refined coordinates rather than at a neighboring continuation
sample while preserving archive compatibility.

`plot_scripts/plot_bd.py` and `plot_scripts/plot_bd_solutions.py`
automatically use these sidecars when present. By default they color
branches by the real unstable-manifold dimension
`unstable_real + 2*unstable_complex_pairs` and mark steady, Hopf, and
multiple transitions separately. Pass `--disable-stability` to recover
ordinary branch styling.

## Legacy Boundary

The following components remain only for models that have not yet been
migrated:

- `IRAM/`;
- `system_operator_Cayley_transform.h`;
- `numerical_algos/arnolid_process/`;
- old model executables that assemble those types directly.

New or migrated models must not assemble those components. They should
instantiate the modern analysis facade with a structured eigensolver
adapter. Legacy files cannot be removed globally until KS2D,
Kolmogorov-flow, overscreening, and periodic-orbit drivers are migrated.

## Required Validation

Before accepting a new model adapter:

1. Test analytical eigenvalues and eigenvectors at known states.
2. Test CPU and every supported device backend from the same source.
3. Check repeated eigenvalues with multiple independent probes.
4. Check projected neutral-mode removal for quotient systems.
5. Run a fresh bifurcation-diagram stability pass.
6. Validate the archive and plotting sidecars with
   `scripts/validate_stability_diagram.py --require-plot-sidecar`.
7. Rerun from the completed archive and verify that persistence is
   unchanged.
