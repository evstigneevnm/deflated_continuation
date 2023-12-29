#include <cmath>
#include <iostream>
#include <cstdio>
#include <memory>
#include <utils/cuda_support.h>
#include <utils/log.h>
#include <external_libraries/cublas_wrap.h>

//problem dependant
#include <nonlinear_operators/circle/circle.h>
#include <nonlinear_operators/circle/linear_operator_circle.h>
#include <nonlinear_operators/circle/preconditioner_circle.h>
#include <nonlinear_operators/circle/convergence_strategy.h>
#include <nonlinear_operators/circle/system_operator.h>
//problem dependant ends
#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/bicgstabl.h>

//problem dependant
#include <common/gpu_vector_operations.h>
#include <common/gpu_file_operations.h>
//problem dependant ends

#include <main/deflation_continuation.hpp>
#include <main/parameters.hpp>


#ifndef Blocks_x_
    #define Blocks_x_ 64
#endif

int main(int argc, char const *argv[])
{
    
    typedef SCALAR_TYPE real;

    size_t Nx = 1; //size of the vector variable. 1 in this case
    
    typedef utils::log_std log_t;
    typedef gpu_vector_operations<real> vec_ops_real;
    typedef gpu_file_operations<vec_ops_real> files_ops_t;
    typedef numerical_algos::lin_solvers::default_monitor<
        vec_ops_real, log_t> monitor_t;

    typedef typename vec_ops_real::vector_type real_vec; 
    typedef nonlinear_operators::circle<
        vec_ops_real, 
        Blocks_x_> circle_t;

    typedef nonlinear_operators::linear_operator_circle<
        vec_ops_real, circle_t> lin_op_t;
    
    typedef nonlinear_operators::preconditioner_circle<
        vec_ops_real, circle_t, lin_op_t> prec_t;


    typedef main_classes::parameters<real> parameters_t;
    parameters_t parameters = main_classes::read_parameters_json<real>("json_project_files/circle_test.json");
    parameters.plot_all();

    Nx = parameters.nonlinear_operator.N_size.at(0)==1?Nx:(throw std::runtime_error("incorrect size for problem in config file provided. Expecting 1."));


    unsigned int m_Krylov = parameters.stability_continuation.Krylov_subspace;
    int nvidia_pci_id = parameters.nvidia_pci_id;
    bool use_high_precision_reduction = parameters.use_high_precision_reduction;

    init_cuda(nvidia_pci_id);
    real norm_wight = std::sqrt(real(Nx));
    real Rad = 1.0;

    auto CUBLAS = std::make_shared<cublas_wrap>();
    vec_ops_real vec_ops_R(Nx, CUBLAS.get() );
    files_ops_t file_ops( (vec_ops_real*) &vec_ops_R);

    circle_t CIRCLE(Rad, Nx, (vec_ops_real*) &vec_ops_R);
    log_t log;
    log_t log_linsolver;
    log_linsolver.set_verbosity(1);
    
    
    //test continuaiton process of a single curve
    typedef main_classes::deflation_continuation<vec_ops_real, files_ops_t, log_t, monitor_t, circle_t, lin_op_t, prec_t, numerical_algos::lin_solvers::bicgstabl, nonlinear_operators::system_operator, parameters_t> deflation_continuation_t;

    deflation_continuation_t DC((vec_ops_real*) &vec_ops_R, (files_ops_t*) &file_ops, (log_t*) &log, (log_t*) &log_linsolver, (circle_t*) &CIRCLE, (parameters_t*) &parameters);

    DC.set_parameters();

    DC.execute();


    return 0;
}
