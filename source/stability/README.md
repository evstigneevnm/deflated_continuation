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

## Model Adapter Contract

`model_adapter_contract.h` defines the compile-time boundary between a
model and the stability implementation. The mandatory interfaces are:

- a linearization provider with
  `set_linearization_point(const vector_type&, scalar_type)`;
- a real operator with
  `apply(const vector_type&, vector_type&)`, returning either `void` or a
  status convertible to `bool`;
- for matrix-free transformed scans, a real affine inverse provider with
  matching `scalar_type`, `vector_type`, and `health_type`, plus
  `apply(jacobian_scale, identity_shift, rhs, solution)` and
  `health(jacobian_scale, identity_shift)`;
- an eigensolver adapter whose `execute(const vector_type&)` returns
  `eigensolver_result<scalar_type>`.

Probe randomization, affine-health detail, classification confirmation,
recycling transactions, and quotient alignment are optional extensions.
The stability layer detects those extensions without making them part of
the nonlinear-operator contract. In particular, models expose only real
arithmetic; complexification remains owned by the transformation layer.

`configure_matrix_free_stability_reuse()` is the shared model-assembly
entry point for Ritz recycling and invariant-subspace tracking. Probe
generators and small-system fallback solvers remain model-specific because
they encode physical constraints and multiplicity information.

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

## Tracked Real Invariant Subspaces

`matrix_free_eigensolver.invariant_subspace_tracking` follows a real
invariant subspace instead of treating each recovered Ritz vector as an
independent scalar object. A real eigenvector contributes one column and
a complex-conjugate eigenpair contributes its real and imaginary plane.
The backend-neutral basis is partitioned into eigenvalue-associated real
blocks and validated at every new linearization point with
`||JQ - Q(Q^T JQ)||_F`. Valid blocks continue to seed probes when another
block has become stale; a complex pair or repeated eigenspace is accepted
or rejected as a block rather than by basis-dependent individual columns.

Validated directions seed additional probes; configured fresh probes are
still executed, so tracking cannot hide a newly appearing eigendirection.
During an independent confirmation transaction, newly recovered columns
are combined with unmatched columns from the previously committed basis.
The unmatched columns are seeded first. This bounded union addresses the
one-sided failure in which one valid Krylov run omits a member of a
multiple or tightly clustered eigenspace.

Principal angles, dimension gaps, validation residuals, and retained-column
counts are reported in the scan diagnostic. Tracker updates commit with the
enclosing classification. Scalar Ritz retries may be reset during transition
refinement without discarding the tracked invariant subspace; both caches are
reset at a true curve-segment boundary.

`maximum_seed_vectors` limits the routine tracking overhead. When every
spectral scan succeeds but the merged physical spectrum still contains fewer
than `aggregation.minimum_eigenpairs`,
`coverage_recovery_maximum_seed_vectors` may request a single transactional
retry with more validated tracked directions. A value of zero disables this
retry. This keeps regular continuation points inexpensive while recovering
tightly clustered or repeated eigenspaces only when the ordinary probes do
not provide the requested coverage.

Independent confirmation runs may fail without aborting immediately. A failed
run consumes `spectrum_classification_retries`, while consensus still requires
`transition_classification_confirmations` successful matching signatures.
Thus solver failures cannot be counted as confirmations, and exhausting the
retry budget still produces an explicit incomplete classification.

If independent runs report different signatures, the matrix-free adapter first
forms a transactional union of their strictly validated physical Ritz values.
Only complete results and aggregate-undercoverage results for which every scan
finished may contribute. Eigenvalues are matched at most once per run, so the
union takes the maximum independently validated multiplicity rather than adding
duplicate discoveries. If that union remains undercovered and the tracked
subspace has accumulated more directions than the nominal seed budget, one
bounded pass with `coverage_recovery_maximum_seed_vectors` augments it.
Otherwise classification remains incomplete. This resolves intermittent
spectral undercoverage without relaxing physical Ritz-residual tolerances or
changing eigensolvers that do not expose reconciliation support.

A successful reconciliation is authoritative rather than another consensus
vote. Its spectrum is assembled transactionally from strict physical Ritz
values at one frozen state, and therefore supersedes compatible incomplete
subsets returned by individual probes. A failed or undercovered reconciliation
does not override the independent-run disagreement.

## Transition-State Recovery

Saved solution vectors can be much farther apart than accepted continuation
points. A straight secant state between two saved vectors may therefore have a
large nonlinear residual, and its Jacobian spectrum is not a branch spectrum.
Transition refinement first uses the configured direct fixed-parameter Newton
correction. If that correction fails, the backend-neutral recovery path marches
from each classified endpoint toward the target parameter. It retries with
`2, 4, ...` fixed-parameter Newton substeps, aligns every accepted state to the
previous symmetry representative, and stops at
`transition_newton_homotopy_maximum_subdivisions`.

The refiner also accepts an optional fallback Newton implementation. A failed
primary correction is transactional: the original state is restored before the
fallback is called, and a failed fallback restores it again. This permits a
model executable to retain its usual continuation solver while supplying a
more robust matrix-free solver for sparse transition-state reconstruction. The
KS2D stability executable uses the NMFD right-preconditioned GMRES settings
from `matrix_free_eigensolver.inner_solver` for this fallback. Complex vectors
and shifts remain confined to the eigensolver; the fallback acts on the real
Jacobian and the ordinary real vector space.

The recovery is controlled by:

- `recover_failed_transition_newton_with_parameter_homotopy`;
- `transition_newton_homotopy_maximum_subdivisions`.

It is invoked only after a direct Newton correction fails. Successful event
diagnostics report the number of homotopy recoveries and accepted substeps.
They also report how many state corrections required the fallback Newton.
This strategy handles smooth fixed-parameter branch segments without changing
the nonlinear operator or vector backend. Before refining a transition between
two saved states, traversal also inspects every archived continuation parameter
between their source indices. A monotone source path uses the ordinary
fixed-parameter refiner. A path with one reversal is split into monotone sides:

- each side is reconstructed transactionally from its saved endpoint;
- the two adjacent turning-point guards are joined in source order with
  forward and reverse secant predictors;
- a smooth turning-point join is accepted only when at least one predicted
  crossing recovers the independently reconstructed state on the other side;
- if neither crossing joins, strict mode rejects the interval, while
  `allow_source_path_topology_splits` preserves it as a topology barrier;
- ordinary transitions are refined independently on each monotone side;
- only a joined dimension change across the adjacent guards is recorded at a local
  quadratic estimate of the parameter extremum, without Newton interpolation
  between the two same-parameter branch states.

Intervals with multiple reversals are decomposed into their complete sequence
of monotone spans. Starting from the saved lower anchor, each span is marched
in continuation source order. Every turning point is crossed with a state
secant predictor and recorded separately when its unstable signature changes.
Classification uses guards displaced from the singular point by
`turning_point_guard_source_points` archived source steps (default: two), while
the fold state and parameter still use the adjacent source points. The
reconstructed path must finish at the saved upper anchor within the configured
numerical matching tolerance;
otherwise the complete curve transaction is rejected. This uses a fixed number
of work vectors independent of the number of turns.

If a forward reconstruction cannot reach the upper anchor, the source-path
marcher also reconstructs backward from that anchor. The reverse path is
aligned to the forward path with the configured finite-symmetry/quotient
aligner. The paths are joined only when their aligned relative state distance
passes the strict numerical matching tolerance. This prevents a successful
fixed-parameter Newton solve on a different branch from being accepted as a
continuation state.

Some legacy curve archives contain a genuine state discontinuity while their
source-point metadata still describes one segment. With
`allow_source_path_topology_splits` disabled, such an interval remains a hard
error. With the option enabled, independently reconstructed paths that do not
meet become an explicit topology barrier instead:

- the last forward state and first aligned reverse state are classified
  independently;
- transition refinement is performed only on each smooth side of the barrier;
- no state or spectrum is interpolated across the barrier;
- both states and their source-point bracket are persisted in
  `debug_curve_stability_topology.dat`;
- the event is exposed to plotting as `topology` and retained as an unresolved
  `source_path_topology` uncertainty for later archive repair.

Topology state files and sidecars are part of the curve transaction. An
aborted curve removes them, and the uncertainty entry is written only after
the stability curve and archive have committed.

The fold work vectors are allocated only while a non-monotone bracket is
processed. The diagnostic replay form is:

```bash
KS2D_stability_cuda.bin config.json dev_num:0 \
  --curve-transition CURVE LOWER_SOURCE UPPER_SOURCE --confirm
```

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

Failed classifications are written atomically to
`stability_uncertainty_registry.json` by default. A record is keyed by the
bifurcation curve, source-point bracket, and failure stage. It stores the
failed parameter, bounded diagnostic text, retry count, and every observed
unstable signature with its occurrence count. The record is not accepted as
stability data. Restart retries the normal curve transaction, and a later
successful point, endpoint confirmation, or transition refinement marks the
same record resolved. Configuration is under
`stability_continuation.classification_uncertainty_registry`.

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

Top-level legacy entry points emit C++ deprecation diagnostics. A legacy
target that cannot yet be changed may define
`STABILITY_SUPPRESS_LEGACY_DEPRECATION_WARNINGS`; this is a compatibility
switch, not permission for new code to depend on the old API.

Migration is model-local: refactor a nonlinear operator to the current
vector/backend contract, add its modern stability adapter and validation,
then retire that model's old assembly. Stability-only ports of legacy
CUDA-specific models are intentionally avoided.

The immutable accepted KS1D and KS2D release identities and numerical
summary counts are pinned in `data/reference/stability_regressions.json`.
`scripts/validate_stability_references.py` also verifies every compact
KS1D replay state byte-for-byte. Updating this lock requires publishing a
new immutable reference asset and reviewing the numerical changes.

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
