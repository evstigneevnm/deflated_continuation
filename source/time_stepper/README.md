# Time Stepping

Supported implementations, tests, and simulations now live in the
[`nmfd-timesteppers`](../contrib/nmfd-timesteppers/README.md) submodule.
Include public headers through `nmfd/time_steppers/...`.

`legacy/` is frozen historical code. It is not used by the refactored timesteppers
and remains unchanged during extraction. Existing parent Make targets and CI jobs
build the submodule sources directly with flat NMFD-operations and SCFD dependencies.
