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

`require_all_scans` and `require_all_probes` establish computational
coverage of the configured work. They do not prove that the selected
shifts enclose every unstable eigenvalue. Shift selection must still be
validated for each model and parameter range.

The probe count must be large enough to expose expected geometric
multiplicity. Constrained models should inject their nonlinear
operator's randomizer instead of relying on the generic vector-space
randomizer.

Dimension changes are confirmed before transition refinement. A
`dimension_guarded_eigensolver` uses its exact host-dense solver for
this confirmation when the state dimension is below the configured
small-system limit, while leaving ordinary branch traversal on the
matrix-free path. When no dedicated confirmation solver is available,
`transition_classification_confirmations` independent matrix-free
classifications must agree. An inconsistent signature aborts the curve
transaction instead of creating a false bifurcation. Confirmed endpoint
signatures also replace provisional regular-point classifications in
the pending stability curve.

`inner_solver.basis_retry_sizes` is an optional strictly increasing
list of GMRES restart dimensions. When a healthy affine factor fails,
the scan retries that factor with each larger basis before trying the
configured shift perturbations. Each retry owns a fresh solver
assembly, so the larger Krylov storage exists only while it is needed.
Physical residual tolerances are not relaxed.

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
