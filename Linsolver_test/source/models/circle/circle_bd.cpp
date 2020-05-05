#include <cmath>
#include <iostream>
#include <cstdio>

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
#include <numerical_algos/lin_solvers/cgs.h>

//problem dependant
#include <common/gpu_vector_operations.h>
#include <common/gpu_file_operations.h>
//problem dependant ends

#include <main/deflation_continuation.hpp>


#ifndef Blocks_x_
    #define Blocks_x_ 64
#endif

int main(int argc, char const *argv[])
{
    
    if(argc!=3)
    {
        printf("Usage: %s dS S\n  dS - continuation step\n   S - number of continuation steps\n",argv[0]);
        return 0;
    }
    typedef SCALAR_TYPE real;

    size_t Nx = 1; //size of the vector variable. 1 in this case
    real dS = atof(argv[1]);
    unsigned int S = atoi(argv[2]);

    
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



    init_cuda(1); // )(PCI) where PCI is the GPU PCI ID
    real norm_wight = std::sqrt(real(Nx));
    real Rad = 1.0;

    //linsolver control
    unsigned int lin_solver_max_it = 1500;
    unsigned int use_precond_resid = 1;
    unsigned int resid_recalc_freq = 1;
    unsigned int basis_sz = 1;
    real lin_solver_tol = 5.0e-3; //relative tolerance wrt rhs vector. For Krylov-Newton method can be set low
    
    //newton control
    unsigned int newton_def_max_it = 350;
    unsigned int newton_def_cont_it = 100;
    real newton_def_tol = 1.0e-9;
    real newton_cont_tol = 1.0e-9;
    real newton_interp_tol = 1.0e-9;


    cublas_wrap *CUBLAS = new cublas_wrap();
    CUBLAS->set_pointer_location_device(false);
    vec_ops_real vec_ops_R(Nx, CUBLAS);
    files_ops_t file_ops( (vec_ops_real*) &vec_ops_R);

    circle_t CIRCLE(Rad, Nx, (vec_ops_real*) &vec_ops_R);
    log_t log;

    
    
    //test continuaiton process of a single curve
    typedef main_classes::deflation_continuation<vec_ops_real, files_ops_t, log_t, monitor_t, circle_t, lin_op_t, prec_t, numerical_algos::lin_solvers::cgs, nonlinear_operators::system_operator> deflation_continuation_t;

    deflation_continuation_t DC((vec_ops_real*) &vec_ops_R, (files_ops_t*) &file_ops, (log_t*) &log, (circle_t*) &CIRCLE );

    DC.set_linsolver(lin_solver_tol, lin_solver_max_it);
    DC.set_extended_linsolver(lin_solver_tol, lin_solver_max_it);
    DC.set_newton(newton_cont_tol, newton_def_cont_it, real(1.0), true);
    DC.set_steps(S, dS);
    
    DC.set_deflation_knots({-1.01, -0.98, -0.95, -0.9, -0.5, -0.1, 0.1, 0.5, 0.95, 0.98, 1.01});

    DC.execute();


    //setup container of curves
//    knots_t deflation_knots;
//    curve_storage_t* curve_storage = new curve_storage_t(vec_ops_R, log, newton, CIRCLE, &deflation_knots);

  
//    delete curve_storage;


    return 0;
}
