#include <cmath>
#include <iostream>
#include <cstdio>
#include <thrust/complex.h>
#include <utils/cuda_support.h>
#include <external_libraries/cufft_wrap.h>
#include <external_libraries/cublas_wrap.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D.h>
#include <numerical_algos/newton_solvers/convergence_strategy.h>
#include <numerical_algos/newton_solvers/newton_solver_extended.h>
#include <utils/log.h>

#include "macros.h"
#include "gpu_file_operations.h"
#include "gpu_vector_operations.h"
#include "test_deflation_typedefs.h"

int main(int argc, char const *argv[])
{


    init_cuda(-1);
    size_t Nx=128;
    size_t Ny=128;




    return 0;
}