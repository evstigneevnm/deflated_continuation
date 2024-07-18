#include <cmath>
#include <iostream>
#include <cstdio>
#include <thrust/complex.h>
#include <common/macros.h>

#include <utils/cuda_support.h>
#include <scfd/utils/trash_cuda_memory.h>
#include <utils/log.h>
#include <external_libraries/cufft_wrap.h>
#include <external_libraries/cublas_wrap.h>

#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/Kuramoto_Sivashinskiy_2D.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/linear_operator_KS_2D.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/preconditioner_KS_2D.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/convergence_strategy.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/system_operator.h>

#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/bicgstabl.h>
#include <numerical_algos/lin_solvers/gmres.h>

#include <numerical_algos/newton_solvers/newton_solver.h>


#include <common/gpu_file_operations.h>
#include <common/gpu_vector_operations.h>


int main(int argc, char const *argv[])
{

    typedef SCALAR_TYPE real;
    typedef thrust::complex<real> complex;
    typedef utils::log_std log_t;
    typedef gpu_vector_operations<real> gpu_vector_operations_real_t;
    typedef gpu_vector_operations<complex> gpu_vector_operations_complex_t;
    typedef gpu_vector_operations<real> gpu_vector_operations_t;
    typedef cufft_wrap_R2C<real> fft_t;
    typedef numerical_algos::lin_solvers::default_monitor<
        gpu_vector_operations_t,log_t> monitor_t;    
    typedef nonlinear_operators::Kuramoto_Sivashinskiy_2D<
        fft_t, 
        gpu_vector_operations_real_t, 
        gpu_vector_operations_complex_t, 
        gpu_vector_operations_t,
        BLOCK_SIZE_X, 
        BLOCK_SIZE_Y> KS_2D;

    typedef nonlinear_operators::linear_operator_KS_2D<
        gpu_vector_operations_t, KS_2D> lin_op_t;
    
    typedef nonlinear_operators::preconditioner_KS_2D<
        gpu_vector_operations_t, KS_2D, lin_op_t> prec_t;

    typedef typename gpu_vector_operations_real_t::vector_type real_vec; 
    typedef typename gpu_vector_operations_complex_t::vector_type complex_vec;
    typedef typename gpu_vector_operations_t::vector_type vec;
    
    using vec_file_ops_t = gpu_file_operations<gpu_vector_operations_t>;
    using real_file_ops_t = gpu_file_operations<gpu_vector_operations_real_t>;

    // using lin_solver_t = numerical_algos::lin_solvers::bicgstabl<lin_op_t,prec_t,gpu_vector_operations_t,monitor_t,log_t>;

    using lin_solver_t = numerical_algos::lin_solvers::gmres<lin_op_t,prec_t,gpu_vector_operations_t,monitor_t,log_t>;

    typedef nonlinear_operators::newton_method::convergence_strategy<
        gpu_vector_operations_t, 
        KS_2D, 
        log_t> convergence_newton_t;
    
    typedef nonlinear_operators::system_operator<
        gpu_vector_operations_t, 
        KS_2D,
        lin_op_t,
        lin_solver_t
        > system_operator_t;
        
    typedef numerical_algos::newton_method::newton_solver<
        gpu_vector_operations_t, 
        KS_2D,
        system_operator_t, 
        convergence_newton_t
        > newton_t;

    
    init_cuda(27);
    scfd::utils::trash_cuda_memory();
    size_t Nx = 512;
    size_t Ny = 512;
    real norm_wight = real(1);//std::sqrt(real(Nx*Ny));
    real size_problem = real(1);//std::sqrt(real(Nx*Ny));

    bool use_high_precision = false;

    //lin_solver control
     unsigned int lin_solver_max_it = 1000;
    real lin_solver_tol = 1.0e-6;
    unsigned int use_precond_resid = 1;
    unsigned int resid_recalc_freq = 1;
    unsigned int basis_sz = 2;
    //newton deflation control
    unsigned int newton_max_it = 350;
    real newton_tol = 5.0e-10;

    real lambda_0 = 4.7;
    real a_val = 2.0;
    real b_val = 4.0;

    fft_t CUFFT_C2R(Nx, Ny);
    size_t My=CUFFT_C2R.get_reduced_size();
    cublas_wrap CUBLAS;
    CUBLAS.set_pointer_location_device(false);

    gpu_vector_operations_real_t vec_ops_R(Nx*Ny, &CUBLAS);
    gpu_vector_operations_complex_t vec_ops_C(Nx*My, &CUBLAS);
    gpu_vector_operations_t vec_ops(Nx*My-1, &CUBLAS);
    
    if(use_high_precision)
    {   vec_ops_R.use_high_precision();
        vec_ops_C.use_high_precision();
        vec_ops.use_high_precision();
    }
    //CUDA GRIDS
    // dim3 Blocks; dim3 Grids; dim3 Grids_F;
    KS_2D KS2D(a_val, b_val, Nx, Ny, &vec_ops_R, &vec_ops_C, &vec_ops, &CUFFT_C2R);
    // KS2D->get_cuda_grid(Grids, Grids_F, Blocks);
    // printf("Blocks = (%i,%i,%i)\n", Blocks.x, Blocks.y, Blocks.z);
    // printf("Grids = (%i,%i,%i)\n", Grids.x, Grids.y, Grids.z);
    // printf("GridsFourier = (%i,%i,%i)\n", Grids_F.x, Grids_F.y, Grids_F.z);
    log_t log;
    log_t log3;
    log3.set_verbosity(0);

    KS_2D* KS2D_p = &KS2D;
    lin_op_t lin_op(KS2D_p);
    prec_t prec(KS2D_p);
    lin_op_t* lin_op_p = &lin_op;
    prec_t* prec_p = &prec;
    gpu_vector_operations_t* vec_ops_p = &vec_ops;
    log_t* log_p = &log;
    monitor_t *mon;
    
    //setup linsolver
    //bcgstabl:
    // lin_solver_t lin_solver(&vec_ops, &log3);
    // lin_solver.set_preconditioner(&prec);
    // lin_solver_t* lin_solver_p = &lin_solver;

    // lin_solver.set_use_precond_resid(use_precond_resid);
    // lin_solver.set_resid_recalc_freq(resid_recalc_freq);
    // lin_solver.set_basis_size(basis_sz);
    // mon = &lin_solver.monitor();
    // mon->init(lin_solver_tol, real(0), lin_solver_max_it);
    // mon->set_save_convergence_history(true);
    // mon->set_divide_out_norms_by_rel_base(true);   
    //gmres:
    lin_solver_t::params params;
    params.basis_size = 100;
    params.preconditioner_side = 'R';
    params.reorthogonalization = true;

    lin_solver_t gmres(&vec_ops, &log3, params);
    gmres.set_preconditioner(&prec);
    mon = &gmres.monitor();
    mon->init(lin_solver_tol, real(0), lin_solver_max_it);
    mon->set_save_convergence_history(true);
    mon->set_divide_out_norms_by_rel_base(true);      
    lin_solver_t* lin_solver_p = &gmres;

    convergence_newton_t conv_newton(vec_ops_p, log_p);
    system_operator_t system_operator(vec_ops_p, lin_op_p, lin_solver_p);
    newton_t newton(vec_ops_p, &system_operator, &conv_newton);
    conv_newton.set_convergence_constants(newton_tol, newton_max_it);

    vec x0;
    vec_ops.init_vector(x0); vec_ops.start_use_vector(x0);
    KS2D.randomize_vector(x0, -1, true);
    bool converged = newton.solve(&KS2D, x0, lambda_0);
    auto solution_norm = vec_ops.norm_l2(x0);
    log.info_f("solution norm = %le", solution_norm);
    if((solution_norm > 1.0e-2)&&(converged))
    {
        KS2D.write_solution("u.pos", x0);
    }

    vec_ops.stop_use_vector(x0); vec_ops.free_vector(x0);


    

    return 0;
}