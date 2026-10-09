# Library Design Guidelines

## Purpose

Rules for new code and regression-tested refactoring, not a list of completed
features. Target matrix-free and sparse CSR/BSR systems on CPU, GPU, and MPI.
Support ODEs, PDEs, DAEs, stationary points (SPs), relative equilibria (REs,
including traveling waves), periodic orbits (POs), and relative periodic orbits
(RPOs).

Changing a backend, discretization, solver, or invariant-object formulation must
not require changes to unrelated algorithm interfaces.

## Core Rules

- Follow the root SCFD-derived `.clang-format` for C/C++ and CUDA code. Use
  clang-format 21 (`make format`, `make format-check`) for reproducible formatting;
  use LF line endings and format touched files before committing. Never format
  vendor/submodule code, generated data, or frozen `source/time_stepper/legacy/` files.
- Use C++17 templates and static polymorphism. Check capabilities with SFINAE,
  `std::void_t`, `if constexpr`, and clear `static_assert` diagnostics.
- Name real-number template parameters `T`, not `Scalar`. Keep established
  interface aliases such as `scalar_type` and `norm_type`.
- Qualify standard-library calls explicitly, e.g. `std::abs`, `std::pow`, and
  `std::isfinite`; do not introduce them with `using std::...` declarations.
- Avoid virtual dispatch, type erasure, and indirect calls in numerical kernels.
  Select runtime implementations at workflow boundaries.
- Generic algorithms consume contracts, never concrete PDE/backend types.
  Keep required methods minimal; do not provide throwing optional-method stubs.
- Implement each mathematical formula once. Residuals, partitions, Jacobians,
  adjoints, and stationary/transient formulations reuse kernels and prepared data.
- Use RAII ownership and explicit non-owning lifetimes. Keep reusable workspaces
  inside their owners, not in public evaluation argument lists.
- Reuse allocations, sparse patterns, FFT plans, and preconditioner resources.
  Avoid hidden copies, transfers, and synchronization; measure performance.
- Use SCFD `operator()` indexing and backend operations for large vector mappings.
  Use direct component mappings for small host systems, not index-based branches
  inside `for_each`. Allow branches only where the algorithm needs them.
- Return structured failures. Never silently treat failed solves as convergence.
  Rejected attempts must not corrupt accepted state or persistent results.
- Keep JSON, CLI parsing, paths, and persistence outside numerical kernels.
- Prefer one problem facade and small internal compositions over extra public
  adapters. Do not add abstractions without a concrete responsibility.

## Dependency Structure

Notation: `X{module}<permitted dependencies>`. Use only needed capabilities;
references to problem modules mean static contracts, not concrete model headers.

```text
A{SCFD backends, arrays, communication, FFT adapters}
B{NMFD vector spaces and operations}<A>
C{NMFD linear-operator contracts and algebraic views}<B>
D{Discretizations}<A,B>
E{Discretized operators and preconditioners, including AMG}<A,B,C,D>
F{NMFD linear solvers and shared Krylov/orthogonalization}<B,C>
G{NMFD nonlinear solvers and globalization}<B,C,F>
H{Symmetry actions, constraints, and state geometry}<B,C,D>
I{Problem-specific nonlinear/evolution facades}<B,C,E,F,G,H>
J{Time integration, scheduling, and event localization}<B,C,F,G,H,I>
K{Flow/Poincare maps, shooting, and space-time problem facades}<B,C,H,I,J>
L{Eigensolvers, invariant subspaces, and Floquet operators}<B,C,F,I,J,K>
M{Deflation}<B,C,G,H,I,K>
N{Continuation}<B,C,F,G,H,I,K>
O{Stability analysis}<B,C,G,H,I,K,L>
P{Neutral archives, topology records, and persistence}<A,B>
Q{Global bifurcation methods}<H,J,K,L,M,N,O,P>
R{Workflow assembly, configuration, visualization, and CI}<I,J,K,M,N,O,P,Q>
```

No sibling-algorithm dependency cycles. Linear solvers receive concrete operators
by template injection. Time integration is a foundation for shooting and Floquet
analysis, not a sibling of deflation/continuation/stability. Space-time spectral
formulations may bypass time integration.

## Vector Spaces And Operators

- Spaces define `scalar_type`, `vector_type`, `norm_type`, lifetime operations,
  assignment, combinations, reductions, norms, and finite-number validation.
- Domain X and codomain Y may differ. Inner products define the adjoint:
  `<Jv,w>_Y = <v,J*w>_X`; weighted adjoints are not plain matrix transposes.
- Use explicit RAII host views, with lightweight host-backend pass-through.
  Never return pointers into shared temporary host buffers.
- Keep high-precision BLAS1, random, complex, and finite-number behavior in
  backend implementations. Maintain matching serial/OMP/CUDA/HIP contracts.
- Linear operators expose `apply(input, output)`; adjoint and parameter actions
  are optional. A lightweight adjoint view presents `apply_adjoint()` as `apply()`
  with exchanged spaces.
- Use `a*A+b*I` only for compatible spaces; otherwise supply the mass/map operator.
  Preconditioners are separate approximate inverses, not convergence controllers.
- Small Hessenberg/Schur matrices use a dense-operation contract. Prefer host
  LAPACK unless device execution is demonstrably beneficial.

## Nonlinear Problems

Each problem exposes one canonical `nonlinear_problem.h`. Small problems may
implement everything there; large problems delegate to private discretization
and solver components. Physical parameters and formulation data are bound
before solving. The existing parameterized entry point is `F(x, parameter, out)`.

A Newton-compatible operator requires:

```cpp
void apply(const vector_type& state, residual_vector_type& residual);
void set_linearization_point(const vector_type& state);
linear_operator_type get_jacobi_operator();
```

`apply()` evaluates the equation being solved: an equilibrium, orbit matching
condition, or implicit stage residual. It is not automatically a physical RHS.
The returned Jacobian exposes `apply(perturbation, result)`; derivative-free
solvers may need only the nonlinear `apply()`.

Optional capabilities include adjoint/parameter actions, preconditioners,
symmetries, exact branches, seed generation, physical output, and consistency
operations. Request only what the selected algorithm uses. Do not mandate
`linear_residual`, `nonlinear_residual`, public workspaces, or separate adapters
for every algorithm.

### Linearization Lifetime

- `set_linearization_point()` prepares the complete derivative of `apply()`.
  Trial residual evaluations must not change this frozen linearization.
- `get_jacobi_operator()` returns a cheap non-owning view, not a copied matrix,
  vector, or workspace. Its owner and backing data must outlive all uses.
- Matrix-free preparation borrows immutable state or caches required data;
  copy only when needed to preserve the base point, into reusable storage.
- Sparse preparation updates coefficients in place. Track revisions explicitly:

| Change | Required action |
| --- | --- |
| Unchanged | Reuse compatible setup. |
| Values changed | Refresh numerical setup; retain valid pattern/symbolic data. |
| Structure/layout changed | Rebuild affected storage and setup. |

Shift, time, state, and distribution changes invalidate dependent views/setup.
Consumers track their own revisions; do not infer updates by comparing vectors.
AMG/factorization reuse needs a validity policy. Do not silently relinearize
through a parameterless `lin.update()`. Concurrent solves use independent mutable
workspaces. Preconditioners consume the complete prepared operator.

## Solver Libraries

- `nmfd-linsolvers`: GMRES/FGMRES, BiCGStab/BiCGStab(L), LSQR/LSMR, shared Arnoldi
  and orthogonalization, monitors, and optional validated subspace recycling.
- Use flexible solvers for changing preconditioners. Validate the original-system
  residual. LSQR/LSMR require forward and adjoint actions; do not form normal
  equations. Preconditioned operators need mathematically consistent adjoints.
- `nmfd-newton` is intended to become `nmfd-nonlinear-solvers`: Newton/inexact
  Newton, LM, adjoint descent, damping, line search, trust regions, and recovery.
- Select directions and globalization independently. Check adjoint availability
  at compile time; never assume a primal preconditioner is its adjoint.
- Coordinate inner tolerances with outer error budgets. Check residuals,
  constraints, finite values, stagnation, and solver status. A small step or
  least-squares stationary point does not certify a root.
- Validate the requested invariant object: an RE satisfies its drift-inclusive
  equations, not necessarily the unprojected stationary equation.

## Time Integration

Refactored timestepper APIs live in `nmfd::time_steppers`, with `runge_kutta`,
`integration`, and `detail` subnamespaces. Unmigrated legacy steppers remain in
`::time_steppers`; new code must use the NMFD namespace. Simple examples for the timesteppers should be incapsulated in a single file and should contain minimum methods required without generating spagetti code or overcomplecated structures.

The structure of the timesteppers is designed as follows:

```cpp
time_step_adaptation // used for adoptation of timestep using embeded methods or other strategies, e.g. constrant time strep, globalization time step adaptation or PID strategy adaptation.
single_step_method // a single step from `n` to `n+1` for explicit or implicit RK, BDF, Rosenbrock or IMEX methods
external_manager // custom external manager that can be used 
continuous_integration // custom class that can return a continuous integration as a polynomial from the selected method for steps `n` to `n+1`.
time_integrator // main class that performs integraton form `t_in` to `t_out` and implements the whole top level logic.
```

All paramters for classes are stored as nested classes, e.g. see `source/contrib/nmfd-newton/include/nmfd/solvers/gmres.h:73 (struct params : public logged_obj_params_t)`.
The user provided problem is given as a class with residual problem in the form:
`M(t,u,p)*udot = f(t,u,p)`
with `M(t,u,p)` is a mass matrix, `udot` is the rate change of vector `u`, `M^{-1}(t,u,p)f(t,u,p)` is the right hand side with the fixed right position of the problem, and `p` is the paramter. The problem is formulated as follows for explicit methds with `f_E` assumed on the right hand side with the corrected sign:
`G(u) = M^{-1}(t,u,p) f_E(t,u_stage,p)`,
so for explicit method, `G` formulated with the following methods:

```cpp
class problem //minimum problem class methods for explicit method. Only one `apply` method  with (in, out)
{
  // ...
  void apply(in, out); // returns `M^{-1} f_E(t,u_stage,p)` as a mapping `in|-G->out` 
  void set_time(t_in); // this for non-autonomous problems, not used for autonomous
  // ...
};
```

for implicit methods:

```math
G(u) = alpha*M(t,u,p)*(u-u_prev) - f_I(t,u,p) - r,
```

Here `f_I` is the full RHS for fully implicit stages, or the implicit part for IMEX, and `G` is formulated with the following methods:

```cpp
class problem //minimum problem class methods for implicit method. Only one `apply` method either with (in, out)
{
  // ...  
  void set_affine_shift(scalar_type alpha); // implements `alpha` affine shift in the residual.
  void apply(const vector_type& in, vector_type& out); // returns the whole residual G(u) as `problem.apply(in, out);` as a mapping `in|-G->out`.
  // void set_previous_state(u_prev); //sets `u_prev` for the calculation of the residual.
  vector_type& previous_state(); // fills the previous state vector `u_prev`
  const vector_type& previous_state() const;
  // void set_known_rhs(r); // sets the known `r` vector that can be substituted for the BDF.
  vector_type& known_rhs(); // fills `r`
  const vector_type& known_rhs() const;
  void set_time(scalar_type time); // this for non-autonomous problems, void for autonomous.
  void set_linearization_point(const vector_type& in); // fixes the point where linearization should be implemented.
  jacobi_operator_type jacobian = get_jacobi_operator(); // returns a thing structure containg the jacobian.
  // ... 
};
```

In minimal configuration jacobian should work as:

```math
DG(u)[v] = alpha*M(u)*v + alpha*(D_u M(u)[v])*(u-u_prev) - D_u f_I(u)[v].
```

and implement:

``` cpp
jacobian.apply(in, out);
```

Other methods are also possible, including:

``` cpp
jacobian.parameter.apply(in, out); // not mandatory
jacobian.adjoint.apply(in, out); // not mandatory
```

The following general status is returned from the timestepper:

```cpp
enum class integration_status
{
    running = 0,
    completed,
    stopped_by_external_operation,
    step_failure,
    external_operation_failure,
    error_estimate_unavailable,
    minimum_step_size_reached,
    attempt_limit_reached,
    rejection_limit_reached,
    step_size_underflow
    // ...
};
```

Additional flags can be added below.

The following minimum class structure should be implemented:

```cpp
namespace nmfd
{
namespace time_steppers
{
template<
  class VectorOperations, 
  class SingleStepMethod, 
  class Log = scfd::utils::log_std, 
  class ExternalManagement = time_steppers::detail::no_external_operations
>
class time_integrator
{
public:
  using scalar_type = typename VectorOperations::scalar_type;
  using vector_type = typename VectorOperations::vector_type;
  // ...
  time_integrator(...): external_manager_stop_(false);
  void set_time_interval(const scalar_type t0, const scalar_type t1) // assuming starts form t0 and integrates to t1.
  void apply(const vector_type& in, vector_type& out) // `in` is the initial vector value and `out` is the final vector value after the integration process is terminated.
  {
    // ... perfrom integration from t0 to t1
    
    if constexpr (!std::is_same_v<ExternalManagement, time_steppers::detail::no_external_operations>) {
      external_manager_.set_time_interval(t_step_in, t_step_out);
      bool modified_state = external_manager_.apply(step_in, step_out); // this aplied the external manager and returns if the state is modified. The stepper should modify the state accordingly with the updated (t_step_out, step_out)
      const auto status = external_manager_.get_status();
      external_manager_stop_ = (status != integration_status::running);
    }
    // ...
  }
  integration_status get_status();
  scalar_type get_final_time();

  // ...
private:
  ExternalManagement external_manager_;
  bool external_manager_stop_;
  // ...
};
}}
```

`no_external_operations` is a dummy class. Custom external manager should have the following minimal interface:

```cpp
namespace nmfd
{
namespace time_steppers
{
class external_manager
{
public:
  // ...
  external_manager(...);
  void set_time_interval(const scalar_type& t_step_in, scalar_type& t_step_out) // sets the time interval between time step, that correspond to step input and step output. 
  bool apply(const vector_type& in, vector_type& out) // can modify the output state. Returns the bool status, if the state was modified in the external manager. 
  integration_status get_status() const // returns the status enum for integration control. 
  // ...
};
}}
```

A single step method is suppled to the `time_integrator` to perform a step advance from `t_step_in` to `t_step_out`. This can be expicit/implcit RK, Rosenbrock/W, BFD or IMEX. The minimal class structure is as follows:

```cpp
namespace nmfd
{
namespace time_steppers
{
namespace detail
{
template<
  class VectorOperations,
  class Problem,
  class TimeStepAdaptation,
  // for implicit methods only:
  class NonlinearSolver 
  // for Rosenbrock/W methods:
  class LinearSolver
>
class single_step_method_{*} // where `{*}` can be `explicit_rk`, `dirk` (this includes sdirk methods), `bdf`, `imex`, `rosenbrock`
{
public:
  // ...
  void set_time(const scalar_type t_step_in); //sets the time to perform time integration
  void set_target_time(scalar_type target_time); 
  void apply(const vector_type& in, vector_type& out); // input and output values after the single step performed
  scalar_type get_dt() const;// returns the timestep used by the single stepper method.
  auto single_step_status = get_status() // returns the status of a single step integrtion results.
  auto continuous_integration = get_continuous_integration() const; // returns a structure for the continuous integration (dense output) of the last step from `n` to `n+1` is a method suports such output. Code structure examples from Numerical Recipes for dense-output steppers.
  void reset(); // clears trajectory-dependent information
  void finalize(adaptation_status outcome, scalar_type committed_time,  const vector_type& committed_state);  //should added to perform modified states and times e.g. from external manager or continuous integraton when the state should be commited. The `apply` method returns only a candidate, the time_integrator finilizes the state. 
};
}}}
```

Support explicit RK, (S)DIRK, Rosenbrock/W, BDF, and IMEX with method-specific engines. IRK and exponential RK are to be supported layer.

The status of a single step is given as the following structure:

```cpp
enum class single_step_status 
{
  converged = 0,
  failed_minimum_dt,
  failed_nan,
  failed_inf,
  failed_solver,
};
```

The `continuous_integration` class is a lightweight structure the has the follwing life time. Dense output describes the pending numerical step and remains valid until the next apply(), reset(), or finalize(). External event processing must finish before finalization. It should provide a structure with:

```cpp
struct continuous_integration 
{
  // ...
  scalar_type evaluate(const scalar_type theta, vector_type& out); // theta in [0, 1] between `n` and `n+1` states, writes state vector at `out` and returns `scalar_type` as the time value tat corrsponds to given `theta`.
  // ...
};
```

Dense output is optional at compile time: disabling it must remove its vector
storage and extra RHS evaluations. RK extension coefficients belong to the table;
evaluation belongs to the step method. Report the actual interpolation order,
including any lower-order fallback. External managers may borrow the step method
to obtain its pending view; the integrator and problem need no extra callbacks.

The `time_step_adaptation` methods will select a particular time step size depending on the local parameters/errors, possible globalization etc.
the general class name is as follows: `time_step_adaptation_{*}`, where `{*}` describes the underlying mechanism in time step adaptation. For example `time_step_adaptation_matlab`, `time_step_adaptation_regulation`, `time_step_adaptation_globalization`, `time_step_adaptation_constant_step` etc.

```cpp
namespace nmfd
{
namespace time_steppers
{
namespace detail
{
class time_step_adaptation
{
public:
  enum class adaptation_status
  {
      accepted = 0,
      accepted_modified,
      rejected,
      failed
  };
  // ...
  scalar_type initialize(const scalar_type time, const vector_type& in, scalar_type dt = static_cast<scalar_type>(-1) ); // sets initial time, vector state value and inital time step (defaulted to -1 for automatic selection.)
  void reset();  // clear adaptation history.
  scalar_type get_dt() const //returns the proposed next step magnitude
  adaptation_status decision = assess(scalar_type time, scalar_type dt_used, const vector_type& u_n, const vector_type& u_np1, scalar_type& dt_next, unsigned char error_order = 0, const vector_type* err_in = nullptr); // `time` - current time used for time step, `dt_used` - time step used for the current step update, `u_n` - solution vector at the previous `n` step, `u_np1` - trial solution vector at the current step `n+1` assuming it is a valid state, `err_in` - estimated state-error vector, already including the step-size factor and can be ignored with defaulted nullptr identifying unavailable error estimate, `dt_next` output: proposed magnitude for the next attempt or next step, `error_order` - the estimate’s scaling exponent for the selected time stepping method, it is optional for methods with no error estimation with default 0.
  void update(adaptation_status outcome, scalar_type time, scalar_type attempted_step_size, const vector_type& current_state); // 
  adaptation_status decision = reject_step(scalar_type dt_used); // triggers reject to a step `dt_used` that returns `decision`.
  // ...
};
}}}
```

Other methods are designed in accordance with the general method algorithms.

## DAEs And Constraints

Use the same operator interface for coupled differential/algebraic residuals.
The mass-form equation above is not a restriction on general DAE residuals.
Do not impose a DAE-index hierarchy or mandatory public `geometry` object.
Declare method applicability; consistent initialization, boundary conditions,
gauges, and hidden constraints remain problem responsibilities.

- Never invert a full singular mass matrix. Explicit evolution needs a valid
  reduced/projected formulation; implicit methods solve the constrained stage.
- For FEM Navier-Stokes, solve `M*udot+N(u)+L*u+G*p=f, D*u=0` with coupled
  velocity/pressure unknowns and a consistent pressure gauge. Vanka, Schur, or
  block preconditioning acts on the complete stage Jacobian.
- Pressure-correction splitting changes the numerical scheme; it is not an
  accepted-step callback or a requirement of a monolithic solve.
- Cahn-Hilliard may use primal or mixed variables, nontrivial mass, and nonlinear
  convex/concave IMEX splitting. Test conservation and energy properties.
- Higher-index problems need appropriate hidden constraints/index reduction.
  Residual scaling must not weaken algebraic-constraint acceptance.

## Symmetry

`problem.symmetry()` optionally declares intrinsic discrete/continuous actions
and generators satisfying `F(g_X*x,p)=g_Y*F(x,p)`. Include only actions preserving
parameters, boundaries, and the discrete space. Mutable charts are algorithm-owned.

| Consumer | Symmetry state |
| --- | --- |
| Storage, output, duplicate detection | Stateless canonical representative or quotient distance. |
| Deflated Newton | Separate chart frozen through each solve. |
| Continuation | Per-semicurve chart; transport state and tangent together. |
| Stability | Independent alignment/linearization state. |
| Time integration | Physical evolution unless quotient dynamics are explicitly selected. |

Support finite-group closure, residual `C_n` actions without vector copies,
isotropy/active-rank changes, slices, and projectors. Physical finite symmetries
and residual slice copies are distinct. Differentiate moving projections fully.
Use identity geometry for ordinary problems; retain valid finite actions for
already reduced problems. Split evolution terms must preserve equivariance.

## MPI And Backends

Keep MPI types/calls behind SCFD communication abstractions. Vector operations
own global reductions, discretizations own halo exchange, FFTs own transposes,
and preconditioners own their exchanges. No public mathematical interface changes.

Workflows inject contexts/subcommunicators; never assume `MPI_COMM_WORLD`.
Check layout/context compatibility and count replicated parameters only once.
Norms, phase conditions, deflation distances, and event decisions are global.
All ranks follow coherent collectives, convergence, retries, and failure paths.
Local exceptions alone cannot recover failed collectives. Distributed adjoints
include reverse communication. Concurrent segments need independent workspaces;
logging/checkpoints must be rank-aware.

## Events, Poincare Sections, And Orbits

Time integration uses generic events and never depends on `periodic_orbit`.
A section supplies `h(t,state)`, crossing orientation/arming, and optional
derivatives. A separate locator brackets and refines intersections.

Use method-specific dense output (`evaluate`, optionally `evaluate_derivative`);
otherwise use bracketed substeps. Interpolation is not an exact trajectory.
For DAEs, reconstruct consistent state/rate through problem-owned operations or
an augmented coupled step/event solve. Validate event and evolution errors
before commit; truncated steps require appropriate history rebuilding.

Single/multiple shooting compose flow/section maps into ordinary nonlinear
systems. Use matrix-free tangent/adjoint maps, including event-time and
consistency derivatives. Near grazing, retain flight times as unknowns and use
bordered section equations. POs need temporal phase conditions; RPOs also need
group actions/spatial phase conditions.

A space-time spectral formulation, including 4D FFT for 3D space plus time,
exposes the same nonlinear interface as shooting. Newton, LM, adjoint descent,
deflation, and continuation must not depend on which formulation is used.

## Deflation, Continuation, And Stability

These are independent reusable modules, assembled only by workflows.

- Deflation composes a residual, solution set, metric, and optional symmetry.
  Keep storage and factor evaluation separate; propagate derivatives correctly:
  `D(dF)[v]=d*DF[v]+Dd[v]*F`. Commit only validated, nonduplicate solutions.
- Continuation accepts neutral seeds from any source. Build pseudo-arclength
  systems through the same nonlinear interface; use `CurveSink` output and
  `BranchQuery` lookup rather than filesystem access.
- Predictor, chart, corrector, and tangent updates are transactional. The
  arclength row uses the applied correction. Knot sampling uses scratch and
  must not silently replace the accepted continuation base.
- Stability consumes linearization, eigensolver, and classification contracts
  through neutral `BranchSource`/`StabilitySink` services. Use the physical
  evolution operator, constrained DAE pencil, or monodromy action; validate
  recovered eigenpairs against it, not merely a transformed operator.
- `source/main/deflation_continuation.hpp` assembles branch discovery/continuation.
  `source/main/stability_continuation.hpp` independently assembles stability
  traversal. Neither reusable algorithm depends on the other workflow.

## Global Bifurcation Roadmap

Build on the same operator/flow/section contracts:
critical-eigenvector branch launching; recurrence-based PO/RPO discovery;
matrix-free Floquet/invariant-subspace analysis; stable/unstable manifold
integration; section-based homoclinic/heteroclinic detection; multiple-shooting
or boundary-value refinement; continuation of connecting orbits; persistent
event/topology graphs. Connections need not be periodic. Finite searches do not
certify branch completeness.

## Persistence And Verification

Use typed configuration at assembly boundaries and versioned, problem-neutral
archives with runtime-rebound paths. Preserve valid partial curves with endpoint
status. Commit/restart transactionally. Keep small CI fixtures in Git and full
reference runs in versioned release assets.

Required regression coverage, scaled to the change:

- Compile-time required/optional capabilities and returned-operator contracts.
- Residual/partition consistency, finite-difference Jacobians, weighted adjoints,
  shifted/state-dependent mass actions, and known-RHS sign/scaling.
- Frozen views, buffer lifetimes, sparse value/pattern revisions, cache reuse,
  rejection rollback, and state-changing callbacks.
- Analytical ODE/DAE and eigenproblem cases; RK coefficients/order conditions,
  error estimators, dense output, constraints, and Poincare event convergence.
- Symmetry equivariance, canonicalization, projectors, charts, and archived
  branch/stability replays.
- CPU/device agreement, high-precision reductions, MPI uneven partitions/global
  decisions, and representative allocation/transfer/operator-call measurements.

Capability-gate CUDA/HIP tests without failing CPU CI when hardware is absent.
Separate long benchmarks from smoke tests. Before accepting a change, check
dependency direction, interface minimality, single-source formulas, ownership,
numerical semantics, and regression evidence.
