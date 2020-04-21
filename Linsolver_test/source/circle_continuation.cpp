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
#include <containers/knots.hpp>
#include <continuation/continuation.hpp>


#ifndef Blocks_x_
    #define Blocks_x_ 64
#endif

int main(int argc, char const *argv[])
{
    
    if(argc!=4)
    {
        printf("Usage: %s lambda_0 dS S\n   lambda_0 - starting parameter\n   dS - continuation step\n   S - number of continuation steps\n",argv[0]);
        return 0;
    }
    typedef SCALAR_TYPE real;

    size_t Nx = 1; //size of the vector variable. 1 in this case
    real lambda0 = atof(argv[1]);
    real dS = atof(argv[2]);
    unsigned int S = atoi(argv[3]);

    
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

    typedef container::knots<real> knots_t;


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
    real newton_def_tol = 1.0e-10;
    real newton_cont_tol = 1.0e-10;
    real newton_interp_tol = 1.0e-10;


    cublas_wrap *CUBLAS = new cublas_wrap();
    CUBLAS->set_pointer_location_device(false);
    vec_ops_real vec_ops_R(Nx, CUBLAS);
    files_ops_t file_ops( (vec_ops_real*) &vec_ops_R);

    circle_t CIRCLE(Rad, Nx, (vec_ops_real*) &vec_ops_R);
    log_t log;
    knots_t deflation_knots;
    deflation_knots.add_element({-1.01, -0.98, -0.95, -0.9, -0.5, -0.1, 0.1, 0.5, 0.95, 0.98, 1.01});
    
    //test continuaiton process of a single curve
    typedef continuation::continuation<vec_ops_real, files_ops_t, log_t, monitor_t, circle_t, lin_op_t, prec_t, knots_t, numerical_algos::lin_solvers::cgs, nonlinear_operators::system_operator> cont_t;

    cont_t cont((vec_ops_real*) &vec_ops_R, (files_ops_t*) &file_ops, (log_t*) &log, (circle_t*) &CIRCLE, (knots_t*) &deflation_knots);

    cont.set_linsolver(lin_solver_tol, lin_solver_max_it);
    cont.set_extended_linsolver(lin_solver_tol, lin_solver_max_it);
    cont.set_newton(newton_cont_tol, newton_def_cont_it, real(1.0), true);
    cont.set_steps(S, dS);
    cont.update_knots();
    

    real_vec x0;
    vec_ops_R.init_vector(x0); vec_ops_R.start_use_vector(x0);
    vec_ops_R.assign_scalar(real(1),x0);

    cont.continuate_curve(x0, lambda0);

    vec_ops_R.stop_use_vector(x0); vec_ops_R.free_vector(x0);

    //setup container of curves
//    knots_t deflation_knots;
//    curve_storage_t* curve_storage = new curve_storage_t(vec_ops_R, log, newton, CIRCLE, &deflation_knots);

  
//    delete curve_storage;


    return 0;
}
