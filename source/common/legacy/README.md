# Legacy Common Operations

These headers and kernels support unmigrated models and their regression tests.
Their relocation changes include paths, not numerical algorithms. Refactored
code uses `nmfd/operations/*` from the NMFD-operations submodule.

- GPU vector/matrix operations and their CUDA implementation files.
- Old reductions, dot products, complex casts, and the vector-operations base.
- Multiprecision matrix operations, vector pool, and the old vector wrapper.
- CPU vector/matrix and GPU matrix file-operation compatibility aliases.
- The unused terminal-color helper.

The common root retains shared scalar math, macros, device initialization,
snapshot storage, output utilities, and compatibility headers still used by
refactored models. The two glued-vector aliases and generic file-operation
header also preserve frozen `source/time_stepper/legacy` include paths.

HYPRE reference code remains only in `source/contrib/nmfd-operations/legacy`.

The legacy GPU-vector implementation does not satisfy the complete modern
operations contract: the shared comparison reports 38 failures involving norms,
sum, and view write-back. Original and relocated implementations produce identical
results. These pre-existing limitations are not changed or hidden by relocation.
