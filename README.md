# deflated_continuation
This is the project for deflated continuation and building skeleton of a bifurcation diagram for a large stationary nonlinear system. This skeleton is then used to identify bifurcation points by an eigenvalue solver.

## C++ Formatting

The root `.clang-format` matches `source/contrib/scfd/.clang-format`.
Use clang-format 21 for consistent results:

```bash
make format
make format-check
```

Set `CLANG_FORMAT=/path/to/clang-format-21` to select the executable. These commands
format project-owned tracked and non-ignored new C/C++ and CUDA files, excluding
vendor/submodule code, generated data, and frozen `source/time_stepper/legacy/`.
The formatting command normalizes source line endings to LF.
