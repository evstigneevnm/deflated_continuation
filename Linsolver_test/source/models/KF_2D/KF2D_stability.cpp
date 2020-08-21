#include <cmath>
#include <iostream>
#include <cstdio>
#include <string>

#include <utils/cuda_support.h>
#include <utils/log.h>
#include <external_libraries/cublas_wrap.h>
#include <external_libraries/cufft_wrap.h>


//problem dependant
#include <common/gpu_vector_operations.h>
#include <common/gpu_matrix_vector_operations.h>
#include <common/gpu_file_operations.h>
//problem dependant ends
//problem dependant
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/Kuramoto_Sivashinskiy_2D.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/linear_operator_KS_2D.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/preconditioner_KS_2D.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/convergence_strategy.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/system_operator.h>
//problem dependant ends

#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/bicgstabl.h>

#include <main/stability.hpp>

int main(int argc, char const *argv[])
{
    
    if(argc!=4)
    {
        printf("For stability analysis:\n");
        printf("Usage: %s path_to_project N m, where:\n",argv[0]);
        printf("    path_to_project is the relative path to the storage of bifurcation diagram data;\n");  
        printf("    N - discretization size in one direction.\n");
        printf("    m - size of the Krylov subspace in Arnoldi process.\n");

        return 0;
    }
    typedef SCALAR_TYPE real;
    std::string path_to_prject_(argv[1]);
    size_t N = atoi(argv[2]);
    size_t m_Krylov = atoi(argv[3]);

    



    return 0;
}