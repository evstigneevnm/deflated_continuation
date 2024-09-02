#include <cmath>
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <thrust/complex.h>
#include <utils/cuda_support.h>
#include <external_libraries/cufft_wrap.h>
#include <external_libraries/cublas_wrap.h>

#include <utils/log.h>
#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/bicgstabl.h>
#include <numerical_algos/lin_solvers/gmres.h>
#include <numerical_algos/lin_solvers/sherman_morrison_linear_system_solve.h>

#include <nonlinear_operators/Kolmogorov_flow_3D/Kolmogorov_3D.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/linear_operator_K_3D.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/preconditioner_K_3D.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/system_operator.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/convergence_strategy.h>
#include <numerical_algos/newton_solvers/newton_solver.h>


#include <containers/knots.hpp>
#include <containers/curve_helper_container.h>
#include <containers/bifurcation_diagram_curve.h>
#include <continuation/continuation.hpp>
#include <deflation/solution_storage.h>

#include <common/macros.h>
#include <common/gpu_file_operations.h>
#include <common/gpu_vector_operations.h>
#include <continuation/single_file_json_params.hpp>


#define Blocks_x_ 32
#define Blocks_y_ 16



int main(int argc, char const *argv[])
{
 
//  lots of typedefs
    using T = SCALAR_TYPE;
    using complex = thrust::complex<T>;
    using gpu_vector_operations_real_t = gpu_vector_operations<T>;
    using gpu_vector_operations_complex_t = gpu_vector_operations<complex>;
    using gpu_vector_operations_t = gpu_vector_operations<T>;
    using gpu_file_operations_t = gpu_file_operations<gpu_vector_operations_t>;
    using cufft_type = cufft_wrap_R2C<T>;
    using KF_3D_t = nonlinear_operators::Kolmogorov_3D<cufft_type, 
            gpu_vector_operations_real_t, 
            gpu_vector_operations_complex_t, 
            gpu_vector_operations_t,
            Blocks_x_, Blocks_y_>;
    using T_vec = typename gpu_vector_operations_real_t::vector_type;
    using complex_vec = typename gpu_vector_operations_complex_t::vector_type;
    using vec = typename gpu_vector_operations_t::vector_type;
    // linear solver config
    using log_t = utils::log_std;
    using monitor_t = numerical_algos::lin_solvers::default_monitor<gpu_vector_operations_t,log_t>;
    using lin_op_t = nonlinear_operators::linear_operator_K_3D<gpu_vector_operations_t, KF_3D_t>;
    using prec_t = nonlinear_operators::preconditioner_K_3D<gpu_vector_operations_t, KF_3D_t, lin_op_t>;    
    using lin_solver_t =  numerical_algos::lin_solvers::bicgstabl
            <
            lin_op_t,
            prec_t,
            gpu_vector_operations_t,
            monitor_t,
            log_t
            >;
    // using lin_solver_t =  numerical_algos::lin_solvers::gmres
    //         <
    //         lin_op_t,
    //         prec_t,
    //         gpu_vector_operations_t,
    //         monitor_t,
    //         log_t
    //         >;            
    using convergence_newton_t =  nonlinear_operators::newton_method::convergence_strategy
            <
            gpu_vector_operations_t, 
            KF_3D_t, 
            log_t
            >;
    
    using system_operator_t = nonlinear_operators::system_operator
            <
            gpu_vector_operations_t, 
            KF_3D_t,
            lin_op_t,
            lin_solver_t
            >;
    using newton_t = numerical_algos::newton_method::newton_solver
            <
            gpu_vector_operations_t,
            KF_3D_t,
            system_operator_t, 
            convergence_newton_t
            >;

    using sherman_morrison_linear_system_solve_t = 
        numerical_algos::sherman_morrison_linear_system::sherman_morrison_linear_system_solve
            <
            lin_op_t,
            prec_t,
            gpu_vector_operations_t,
            monitor_t,
            log_t,
            // numerical_algos::lin_solvers::bicgstabl
            numerical_algos::lin_solvers::gmres
            >;
    using knots_t = container::knots<T>;
    using container_helper_t = container::curve_helper_container<gpu_vector_operations_t>;
    using sol_storage_def_t = deflation::solution_storage<gpu_vector_operations_t>;  
    using bif_diag_curve_t = container::bifurcation_diagram_curve
            <
            gpu_vector_operations_t,
            gpu_file_operations_t, 
            log_t,
            KF_3D_t,
            newton_t, 
            sol_storage_def_t,
            container_helper_t
            >;
    using continuate_t = continuation::continuation
            <
            gpu_vector_operations_t, 
            gpu_file_operations_t, 
            log_t, 
            KF_3D_t, 
            lin_op_t,  
            knots_t,
            sherman_morrison_linear_system_solve_t,  
            newton_t,
            bif_diag_curve_t
            >;
//  typedefs ends



    // if(argc != 11)
    // {
    //     std::cout << argv[0] << " Steps R_min R_max file_name alpha R N high_prec direction dS:\n    Steps - number of continuation steps;\n    R_min, R_max - minimum and maximum Reynolds numbers;\n    file_name - name of the file with an initial solution;\n    0<alpha<=1 - torus stretching parameter;\n    R is the Reynolds number that corresponds to the file_name solution;\n    N - discretization in one direction;\n    high_prec = (0/1) usage of high precission reduction methods;\n    direction = (-1/1) - direction of continuation.\n    dS - step size of the continuaiton method. Not use continuaiton, if dS<0\n";
    //     return(0);       
    // }
    
    // ./exec_file_name.bin json_file_name nz file_name R_value direction
    if(argc != 7)
    {
        std::cout << argv[0] << " json_file_name nz file_name R_value direction save_folder:" << std::endl;
        std::cout <<  "  json_file_name - configuration file for the problem," << std::endl;
        std::cout <<  "  nz - forcing in Z direction: 0,1,2...," << std::endl;
        std::cout <<  "  file_name - path/file_name to the starting file," << std::endl;
        std::cout <<  "  R_value - parameter value for the starting file," << std::endl;
        std::cout <<  "  direction - direction on which to start continuation 1/-1." << std::endl;
        std::cout <<  "  save_folder - where the results of continuation will be stored." << std::endl;
        return(0);          
    }

    std::string json_config_name(argv[1]);
    int nz = std::stoi(argv[2]);
    std::string file_name(argv[3]);
    T R = std::stof(argv[4]);
    int direction = std::stoi(argv[5]);
    std::string folder(argv[6]);
    auto params = file_params::read_parameters_json(json_config_name);



    int steps = params.steps;
    T R_min = params.R_min;
    T R_max = params.R_max;
    size_t N = params.N;
    auto alpha = params.alpha;
    int one_over_alpha = int(1.0/alpha);
    int high_prec = params.high_prec;
    auto dS_0 = params.dS_0;
    auto dSmax = params.dSmax;
    auto dSdec_mult = params.dSdec_mult;
    auto dSinc_mult = params.dSinc_mult;    
    unsigned int skip_output = params.skip_output;


//  parameters for solvers
    //linsolver control
    unsigned int lin_solver_max_it = params.solvers_config.linear_solver.max_iterations;
    T lin_solver_tol = params.solvers_config.linear_solver.tolerance;
    unsigned int use_precond_resid = params.solvers_config.linear_solver.use_precond_resid;
    unsigned int resid_recalc_freq = params.solvers_config.linear_solver.resid_recalc_freq;
    unsigned int basis_sz = params.solvers_config.linear_solver.basis_sz;
    //extended linsolver control
    unsigned int lin_solver_max_it_ex = params.solvers_config.linear_solver_extended.max_iterations;
    T lin_solver_tol_ex = params.solvers_config.linear_solver_extended.tolerance;
    unsigned int use_precond_resid_ex = params.solvers_config.linear_solver_extended.use_precond_resid;
    unsigned int resid_recalc_freq_ex = params.solvers_config.linear_solver_extended.resid_recalc_freq;
    unsigned int basis_sz_ex = params.solvers_config.linear_solver_extended.basis_sz;    
    //newton continuation control
    unsigned int newton_cont_max_it = params.solvers_config.newton_method.max_iterations;
    T newton_cont_tol = params.solvers_config.newton_method.tolerance;
    //exended newton continuation control
    unsigned int newton_cont_maximum_iterations =  params.solvers_config.newton_method_extended.max_iterations;
    T newton_cont_update_wight_maximum = params.solvers_config.newton_method_extended.update_wight_maximum;
    bool newton_cont_save_norms_history = params.solvers_config.newton_method_extended.save_norms_history;
    bool newton_cont_verbose  = params.solvers_config.newton_method_extended.verbose;
    T newton_cont_tolerance = params.solvers_config.newton_method_extended.tolerance;
    T newton_cont_relax_tolerance_factor = params.solvers_config.newton_method_extended.relax_tolerance_factor;
    unsigned int newton_cont_relax_tolerance_steps = params.solvers_config.newton_method_extended.relax_tolerance_steps;
    unsigned int newton_stagnation_p_max = params.solvers_config.newton_method_extended.stagnation_p_max;
    T newton_maximum_norm_increase = params.solvers_config.newton_method_extended.maximum_norm_increase;



    init_cuda(-1);
    size_t Nx = N*one_over_alpha;
    size_t Ny = N;
    size_t Nz = N;

    std::cout << "Using: Steps = " << steps << ", Rmin = " << R_min << ", Rmax = " << R_max << std::endl;
    std::cout << "    file name = " << file_name << ", alpha = " << alpha << ", R = " << R << std::endl;
    std::cout << "    Nx X Ny X Nz = " << Nx <<" X " << Ny << " X " << Nz << std::endl;
    std::cout << "    high prec: " << (high_prec==1?"yes":"no") << ", direction: " << (direction==1?"'+'":"'-'") << ", dS = " << (dS_0<T(0.0)?-1:dS_0) << std::endl;
    std::cout << "    nz = " << nz << std::endl;

    //seting all vector operations and low level libraries
    cufft_type *CUFFT_C2R = new cufft_type(Nx, Ny, Nz);
    size_t Mz = CUFFT_C2R->get_reduced_size();
    cublas_wrap *CUBLAS = new cublas_wrap(true);
    CUBLAS->set_pointer_location_device(false);
    
    gpu_vector_operations_real_t *vec_ops_R = new gpu_vector_operations_real_t(Nx*Ny*Nz, CUBLAS);
    gpu_vector_operations_complex_t *vec_ops_C = new gpu_vector_operations_complex_t(Nx*Ny*Mz, CUBLAS);
    size_t N_global = 3*(Nx*Ny*Mz-1);
    gpu_vector_operations_t *vec_ops = new gpu_vector_operations_t(N_global, CUBLAS);
    if(high_prec == 1)
    {
        vec_ops_R->use_high_precision();
        vec_ops_C->use_high_precision();
        vec_ops->use_high_precision();
    }
    gpu_file_operations_t *file_ops = new gpu_file_operations_t(vec_ops);
    KF_3D_t *KF_3D = new KF_3D_t(alpha, Nx, Ny, Nz, vec_ops_R, vec_ops_C, vec_ops, CUFFT_C2R, true, nz);    
    //monitor control
    monitor_t *mon;
    log_t log;
    log_t log3;
    log3.set_verbosity(0);
    lin_op_t lin_op(KF_3D);
    prec_t prec(KF_3D);      


    lin_solver_t lin_solver(vec_ops, &log3);
    lin_solver.set_preconditioner(&prec);
    lin_solver.set_use_precond_resid(use_precond_resid);
    lin_solver.set_resid_recalc_freq(resid_recalc_freq);
    lin_solver.set_basis_size(basis_sz);
    mon = &lin_solver.monitor();
    mon->init(lin_solver_tol, T(0.0), lin_solver_max_it);
    mon->set_save_convergence_history(true);
    mon->set_divide_out_norms_by_rel_base(true);   
    knots_t knots;
    
    knots.add_element( std::vector<T>({R_min, R, R_max}) );


    //solutions vector
    vec x; vec_ops->init_vector(x); vec_ops->start_use_vector(x);
    vec Fx; vec_ops->init_vector(Fx); vec_ops->start_use_vector(Fx);

    log_t *log_p = &log;
    convergence_newton_t *conv_newton = new convergence_newton_t(vec_ops, log_p);
    
    lin_op_t* lin_op_p = &lin_op;
    lin_solver_t* lin_solver_p = &lin_solver;
    system_operator_t *system_operator = new system_operator_t(vec_ops, lin_op_p, lin_solver_p);
    newton_t *newton = new newton_t(vec_ops, system_operator, conv_newton);
    conv_newton->set_convergence_constants(newton_cont_tol, newton_cont_max_it, 1.0, true, true);
    
    //set up continuation method
    sherman_morrison_linear_system_solve_t SM(&prec, vec_ops, &log3);
    
    mon = &SM.get_linsolver_handle()->monitor();
    mon->init(lin_solver_tol, T(0), lin_solver_max_it_ex);
    mon->set_save_convergence_history(true);
    mon->set_divide_out_norms_by_rel_base(true);
    SM.get_linsolver_handle()->set_use_precond_resid(use_precond_resid_ex);
    SM.get_linsolver_handle()->set_resid_recalc_freq(resid_recalc_freq_ex);
    SM.get_linsolver_handle()->set_basis_size(basis_sz_ex);

    container_helper_t container_helper(vec_ops);
    sol_storage_def_t solution_storage(vec_ops, 1, 1.0);
 
    int curve_number = 1;
    bif_diag_curve_t bif_diag(vec_ops, file_ops, &log, KF_3D, newton, curve_number, folder,  &container_helper, skip_output);


    continuate_t continuate(vec_ops, file_ops, (log_t*) &log, KF_3D, lin_op_p, &knots, &SM, newton);

    continuate.set_newton(newton_cont_tolerance, newton_cont_maximum_iterations, newton_cont_relax_tolerance_factor, newton_cont_relax_tolerance_steps, newton_cont_update_wight_maximum, newton_cont_save_norms_history, newton_cont_verbose, newton_stagnation_p_max, newton_maximum_norm_increase);
    continuate.set_steps(steps, dS_0, dSmax, direction, dSdec_mult, dSinc_mult, 5);

    if (file_name!="none")
    {
        printf("reading file %s\n", file_name.c_str() );
        file_ops->read_vector(file_name, x);

    }
    else
    {
        printf("Generating a random vector\n");
        KF_3D->randomize_vector(x);
        file_ops->write_vector("dat_files/rand_vec.dat", x);

    }
    printf("initial div = %le\n",double(KF_3D->div_norm(x) ));
    KF_3D->write_solution_abs("x0_abs.pos", x);
    KF_3D->write_solution_vec("x0_vec.pos", x);
    if( dS_0>T(0.0) )
    {
        KF_3D->project(x);
    }

    bool converged = newton->solve(KF_3D, x, R);

    if(!converged)
    {
        printf("Newton method for %s failed to converge!\n", file_name.c_str() );
    }
    KF_3D->F(x, R, Fx);
    printf("Newton method converged for %s with solution norm = %le, div = %le, ||F|| = %le\n", file_name.c_str(), double(vec_ops->norm_l2(x) ), double(KF_3D->div_norm(x) ), double(vec_ops->norm(Fx))  );
    KF_3D->write_solution_abs("x_abs.pos", x);
    KF_3D->write_solution_vec("x_vec.pos", x);

    if(dS_0 > T(0.0) )
    {
        if(converged)
        {
            bif_diag_curve_t* bdf_p = &bif_diag;
            continuate.continuate_curve(bdf_p, x, R);
        }
        else
        {
            printf("continuation cannot be performed since the solution didn't converge.\n");
        }
    }



    vec_ops->stop_use_vector(x); vec_ops->free_vector(x);
    vec_ops->stop_use_vector(Fx); vec_ops->free_vector(Fx);
    delete newton;
    delete system_operator;
    delete conv_newton;
    

    delete file_ops;
    delete KF_3D;
    delete vec_ops_R;
    delete vec_ops_C;
    delete vec_ops;
    delete CUFFT_C2R;
    delete CUBLAS;

    return 0;
}