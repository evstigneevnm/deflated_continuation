#include <cmath>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <thrust/complex.h>
#include <utils/cuda_support.h>
#include <external_libraries/cufft_wrap.h>
#include <external_libraries/cublas_wrap.h>
#include <external_libraries/lapack_wrap.h>

#include <utils/log.h>
#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/bicgstabl.h>

#include <nonlinear_operators/Kolmogorov_flow_3D/Kolmogorov_3D.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/linear_operator_K_3D.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/preconditioner_K_3D.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/linear_operator_K_3D_shifted.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/preconditioner_K_3D_shifted.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/system_operator.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/convergence_strategy.h>

#include <numerical_algos/newton_solvers/newton_solver.h>

#include <common/macros.h>
#include <common/gpu_file_operations.h>
#include <common/gpu_vector_operations.h>
#include <common/gpu_matrix_vector_operations.h>

#include <numerical_algos/arnolid_process/arnoldi_process.h>
#include <stability/system_operator_Cayley_transform.h>
#include <stability/IRAM/iram_process.hpp>
#include <stability/stability_analysis.hpp>

#include <time_stepper/explicit_time_step.h>
#include <time_stepper/time_stepper.h>


#define Blocks_x_ 32
#define Blocks_y_ 16



int main(int argc, char const *argv[])
{
 
    using real = SCALAR_TYPE;
    using complex = thrust::complex<real>;
    using gpu_vector_operations_real_t = gpu_vector_operations<real>;
    using gpu_vector_operations_complex_t = gpu_vector_operations<complex>;
    using gpu_vector_operations_t = gpu_vector_operations<real>;
    using gpu_file_operations_t = gpu_file_operations<gpu_vector_operations_t>;
    using cufft_type = cufft_wrap_R2C<real>;
    using KF_3D_t = nonlinear_operators::Kolmogorov_3D<cufft_type, 
            gpu_vector_operations_real_t, 
            gpu_vector_operations_complex_t, 
            gpu_vector_operations_t,
            Blocks_x_, Blocks_y_>;
    using real_vec = typename gpu_vector_operations_real_t::vector_type; 
    using complex_vec = typename gpu_vector_operations_complex_t::vector_type;
    using vec = typename gpu_vector_operations_t::vector_type;
   
    using gpu_matrix_vector_operations_t = gpu_matrix_vector_operations<real, vec>;
    using mat = typename gpu_matrix_vector_operations_t::matrix_type;
    using lapack_wrap_t = lapack_wrap<real>;

    if(argc != 9)
    {
        std::cout << argv[0] << " file_1 file_2 alpha R1 R2 N m k:\n 0<alpha<=1, R_j is the j-th Reynolds number (between two points),\n   N = 2^n- discretization in one direction, \n   m is the total number of Krylov basis vectors for the eigensolver (additional vectors = m - k)\n   k is the number of Krylov basis vectors for the eigensolver.\n";
        return(0);       
    }
    
    std::string file_1(argv[1]);
    std::string file_2(argv[2]);
    real alpha = std::atof(argv[3]);
    real Rey = std::atof(argv[4]);
    real Rey2 = std::atof(argv[5]);
    size_t N = std::atoi(argv[6]);
    size_t m = std::atoi(argv[7]);
    size_t k = std::atoi(argv[8]);
    int one_over_alpha = int(1/alpha);


    init_cuda(9);
    size_t Nx = N*one_over_alpha;
    size_t Ny = N;
    size_t Nz = N;
    std::cout << "Using alpha = " << alpha << ", Reynolds 1 = " << Rey << ", Reynolds 2 = " << Rey2 << ", with discretization: " << Nx << "X" << Ny << "X" << Nz << ", dim(Krylov) = " << m << " and desired eigenvalues = " << k << std::endl;
    if(file_1 != "-")
    {
        std::cout << "file 1: " << file_1 << " file 2: " << file_2 << std::endl;    
    }
    else
    {
        std::cout << "files are not provided, using analytical solutions from the nonlinear operator" << std::endl;
    }


    //linsolver control
    unsigned int lin_solver_max_it = 1000;
    real lin_solver_tol = 5.0e-4;
    unsigned int use_precond_resid = 1;
    unsigned int resid_recalc_freq = 1;
    unsigned int basis_sz = 3;
    //newton deflation control
    unsigned int newton_def_max_it = 250;
    real newton_def_tol = 1.0e-9;


    lapack_wrap<real> lapack(m);
    cufft_type *CUFFT_C2R = new cufft_type(Nx, Ny, Nz);
    size_t Mz = CUFFT_C2R->get_reduced_size();
    cublas_wrap *CUBLAS = new cublas_wrap(true);
    CUBLAS->set_pointer_location_device(false);
    
    gpu_vector_operations_real_t *vec_ops_R = new gpu_vector_operations_real_t(Nx*Ny*Nz, CUBLAS);
    gpu_vector_operations_complex_t *vec_ops_C = new gpu_vector_operations_complex_t(Nx*Ny*Mz, CUBLAS);
    size_t N_global = 3*(Nx*Ny*Mz-1);
    
    gpu_vector_operations_t *vec_ops = new gpu_vector_operations_t(N_global, CUBLAS);
    gpu_matrix_vector_operations_t *mat_ops = new gpu_matrix_vector_operations_t(N_global, m, CUBLAS);
    gpu_vector_operations_t *vec_ops_small = new gpu_vector_operations_t(m, CUBLAS);
    gpu_matrix_vector_operations_t *mat_ops_small = new gpu_matrix_vector_operations_t(m, m, CUBLAS);
    gpu_file_operations_t *file_ops = new gpu_file_operations_t(vec_ops);



    KF_3D_t *KF_3D = new KF_3D_t(alpha, Nx, Ny, Nz, vec_ops_R, vec_ops_C, vec_ops, CUFFT_C2R);
    // linear operators and solvers config
    using log_t = utils::log_std;
    using monitor_t = numerical_algos::lin_solvers::default_monitor<
        gpu_vector_operations_t,log_t>;
    using lin_op_t = nonlinear_operators::linear_operator_K_3D<
        gpu_vector_operations_t, KF_3D_t>;
    using lin_op_shifted_t = nonlinear_operators::linear_operator_K_3D_shifted<gpu_vector_operations_t, KF_3D_t>;
    using prec_t = nonlinear_operators::preconditioner_K_3D<
        gpu_vector_operations_t, KF_3D_t, lin_op_t>;
    using prec_shifted_t = nonlinear_operators::preconditioner_K_3D_shifted<gpu_vector_operations_t, KF_3D_t, lin_op_shifted_t>;
    using lin_solver_t = numerical_algos::lin_solvers::bicgstabl<
        lin_op_t,prec_t,gpu_vector_operations_t,monitor_t,log_t>;
    using lin_solver_shifted_t = numerical_algos::lin_solvers::bicgstabl<
        lin_op_shifted_t,prec_shifted_t,gpu_vector_operations_t,monitor_t,log_t>;

//  eigensolver config
    using Cayley_system_op_t = stability::system_operator_Cayley_transform<gpu_vector_operations_t, KF_3D_t, lin_op_shifted_t, lin_solver_shifted_t, log_t>;
    using arnoldi_t = numerical_algos::eigen_solvers::arnoldi_process<gpu_vector_operations_t, gpu_matrix_vector_operations_t, Cayley_system_op_t, log_t>;
    using iram_t = stability::IRAM::iram_process<gpu_vector_operations_t, gpu_matrix_vector_operations_t, lapack_wrap_t, arnoldi_t, Cayley_system_op_t, lin_op_t, log_t>;


    // newton with convergence strategy config
    using convergence_newton_t = nonlinear_operators::newton_method::convergence_strategy<gpu_vector_operations_t, KF_3D_t, log_t>;
    using system_operator_t = nonlinear_operators::system_operator<
        gpu_vector_operations_t, KF_3D_t, lin_op_t, lin_solver_t>;
    using newton_t = numerical_algos::newton_method::newton_solver<
        gpu_vector_operations_t, KF_3D_t, system_operator_t, convergence_newton_t>;
    // main stability analysis config
    using stability_analysis_t = stability::stability_analysis<gpu_vector_operations_t, KF_3D_t, log_t, newton_t, iram_t>;



    monitor_t *mon;

    log_t log;
    log_t log3;
    log3.set_verbosity(0);
    lin_op_t lin_op(KF_3D);
    prec_t prec(KF_3D);
    lin_op_shifted_t lin_op_sh(vec_ops, KF_3D);
    prec_shifted_t prec_sh(KF_3D);

    lin_solver_t lin_solver(vec_ops, &log);
    lin_solver.set_preconditioner(&prec);

    lin_solver_shifted_t lin_solver_sh(vec_ops, &log3);
    lin_solver_sh.set_preconditioner(&prec_sh);

    lin_solver.set_use_precond_resid(use_precond_resid);
    lin_solver.set_resid_recalc_freq(resid_recalc_freq);
    lin_solver.set_basis_size(basis_sz);
    mon = &lin_solver.monitor();
    mon->init(lin_solver_tol, real(0.0), lin_solver_max_it);
    mon->set_save_convergence_history(true);
    mon->set_divide_out_norms_by_rel_base(true);

    lin_solver_sh.set_use_precond_resid(use_precond_resid);
    lin_solver_sh.set_resid_recalc_freq(resid_recalc_freq);
    lin_solver_sh.set_basis_size(basis_sz);
    mon = &lin_solver_sh.monitor();
    mon->init(lin_solver_tol, real(0.0), lin_solver_max_it);
    mon->set_save_convergence_history(true);
    mon->set_divide_out_norms_by_rel_base(true);


    vec x0, x1, dx, x_back, b;
    vec x0_st, x1_st; //for timestepper

    vec_ops->init_vector(b); vec_ops->start_use_vector(b);
    vec_ops->init_vector(x0); vec_ops->start_use_vector(x0);
    vec_ops->init_vector(x1); vec_ops->start_use_vector(x1);
    vec_ops->init_vector(dx); vec_ops->start_use_vector(dx);
    vec_ops->init_vector(x_back); vec_ops->start_use_vector(x_back);
    vec_ops->init_vector(x0_st); vec_ops->start_use_vector(x0_st);
    vec_ops->init_vector(x1_st); vec_ops->start_use_vector(x1_st);    




    log_t *log_p = &log;
    convergence_newton_t *conv_newton = new convergence_newton_t(vec_ops, log_p);
    lin_op_t* lin_op_p = &lin_op;
    lin_op_shifted_t* lin_op_sh_p = &lin_op_sh;
    lin_solver_shifted_t* lin_solver_sh_p = &lin_solver_sh;
    lin_solver_t* lin_solver_p = &lin_solver;
    system_operator_t *system_operator = new system_operator_t(vec_ops, lin_op_p, lin_solver_p);
    newton_t *newton = new newton_t(vec_ops, system_operator, conv_newton);


    //configuring eigensolver:
    Cayley_system_op_t Cayler_sys_op(vec_ops, KF_3D, lin_op_sh_p, lin_solver_sh_p, log_p);
    Cayler_sys_op.set_tolerance(1.0e-9);
    
    real sigma = 10.0;
    real mu = -10.0;
    Cayler_sys_op.set_sigma_and_mu(sigma, mu);

    arnoldi_t arnoldi(vec_ops, vec_ops_small, mat_ops, mat_ops_small, &Cayler_sys_op, log_p);
    iram_t iram(vec_ops, mat_ops, vec_ops_small, mat_ops_small, &lapack, &arnoldi, &Cayler_sys_op, lin_op_p, log_p);
    iram.set_verbocity(true);
    iram.set_target_eigs("LR");
    iram.set_number_of_desired_eigenvalues(k);
    iram.set_tolerance(1.0e-6);
    iram.set_max_iterations(100);
    iram.set_verbocity(true);

    stability_analysis_t *stabl = new stability_analysis_t(vec_ops, log_p, KF_3D, newton, &iram);

    conv_newton->set_convergence_constants(newton_def_tol, newton_def_max_it, 1.0, true, true);

    
    if(file_1 != "-")
    {
        file_ops->read_vector(file_1, x0);
        bool converged = newton->solve(KF_3D, x0, Rey);
        if(!converged)
        {
            printf("Newton file 1 failed to converge!\n");
        }
        printf("Newton file 1 solution norm = %le, div = %le\n", vec_ops->norm_l2(x0), KF_3D->div_norm(x0));
        KF_3D->write_solution_abs("x_1.pos", x0);


        stabl->execute(x0, Rey);

    //  testing bisection
        file_ops->read_vector(file_2, x1);
        //file_ops->read_vector("x_2.dat", x1);

        bool converged2 = newton->solve(KF_3D, x1, Rey2);
        if(!converged2)
        {
            printf("Newton file 2 failed to converge for the second Reynolds number!\n");
        }    
        printf("Newton file 2 solution norm = %le, div = %le\n", vec_ops->norm_l2(x1), KF_3D->div_norm(x1));
        KF_3D->write_solution_abs("x_2.pos", x1);
        //file_ops->write_vector("x_2.dat", x1);

        real lambda_p = 0.0;
        stabl->bisect_bifurcaiton_point(x0, Rey, x1, Rey2, dx, lambda_p);
        std::cout <<  "Bisected parameter value = " << lambda_p << std::endl;
        printf("Bisected solution norm = %le, div = %le\n", vec_ops->norm_l2(dx), KF_3D->div_norm(dx));
        KF_3D->write_solution_abs("x_bisected.pos", dx);
    }
    else
    {
        real simulation_time = 10.0;
        using time_step_t = time_steppers::explicit_time_step<gpu_vector_operations_t, KF_3D_t, log_t>;
        using time_stepper_t = time_steppers::time_stepper<gpu_vector_operations_t, KF_3D_t, time_step_t,log_t>;
        time_step_t explicit_step(vec_ops, KF_3D, log_p);
        time_stepper_t time_stpr(vec_ops, KF_3D, &explicit_step, log_p);
        explicit_step.set_time_step(5.0e-3);

        KF_3D->exact_solution(Rey, x0);
        KF_3D->write_solution_abs("x_1a.pos", x0);
        KF_3D->write_solution_vec("x_1v.pos", x0);
        stabl->execute(x0, Rey);
        
        log.info_f("executing time stepper with time = %.2e", simulation_time);
        time_stpr.set_parameter(Rey);
        time_stpr.set_time(50.0);
        time_stpr.set_initial_conditions(x0, 1.0e-2);
        time_stpr.get_results(x0);
        KF_3D->write_solution_abs("x_1a_pert.pos", x0);
        time_stpr.execute();
        time_stpr.save_norms("probe_1.dat");
        time_stpr.get_results(x0);
        KF_3D->write_solution_abs("x_1a_sim.pos", x0);        

        time_stpr.reset();

        KF_3D->exact_solution(Rey2, x1);
        KF_3D->write_solution_abs("x_2a.pos", x1);
        KF_3D->write_solution_vec("x_2v.pos", x1);
        stabl->execute(x1, Rey2);
        
        log.info_f("executing time stepper with time = %.2e", simulation_time);
        time_stpr.set_parameter(Rey2);
        time_stpr.set_time(1500.0);
        time_stpr.set_initial_conditions(x1, 1.0e-2);
        time_stpr.get_results(x1);
        KF_3D->write_solution_abs("x_2a_pert.pos", x1);
        time_stpr.execute();
        time_stpr.save_norms("probe_2.dat");   
        time_stpr.get_results(x1);
        KF_3D->write_solution_abs("x_2a_sim.pos", x1);             
    }

    vec_ops->stop_use_vector(b); vec_ops->free_vector(b);
    vec_ops->stop_use_vector(x0); vec_ops->free_vector(x0);
    vec_ops->stop_use_vector(x1); vec_ops->free_vector(x1);
    vec_ops->stop_use_vector(dx); vec_ops->free_vector(dx);
    vec_ops->stop_use_vector(x_back); vec_ops->free_vector(x_back);
    vec_ops->stop_use_vector(x0_st); vec_ops->free_vector(x0_st);
    vec_ops->stop_use_vector(x1_st); vec_ops->free_vector(x1_st);

    delete stabl;
    delete newton;
    delete system_operator;
    delete conv_newton;
    

    delete file_ops;
    delete KF_3D;
    delete vec_ops_R;
    delete vec_ops_C;
    delete vec_ops;
    delete vec_ops_small;
    delete mat_ops;
    delete mat_ops_small;
    delete CUFFT_C2R;
    delete CUBLAS;

    return 0;
}