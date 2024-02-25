#include <cmath>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sstream>
#include <thrust/complex.h>
#include <utils/cuda_support.h>

#include <scfd/utils/init_cuda.h>
#include <external_libraries/cufft_wrap.h>
#include <external_libraries/cublas_wrap.h>

#include <scfd/utils/log.h>
#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/bicgstabl.h>
#include <numerical_algos/lin_solvers/exact_wrapper.h>

#include <nonlinear_operators/Kolmogorov_flow_3D/Kolmogorov_3D.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/linear_operator_K_3D.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/preconditioner_K_3D.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/system_operator.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/convergence_strategy.h>

#include <nonlinear_operators/Kolmogorov_flow_3D/linear_operator_K_3D_stiff.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/preconditioner_K_3D_stiff.h>


#include <common/macros.h>
#include <common/gpu_file_operations.h>
#include <common/gpu_vector_operations.h>


int main(int argc, char const *argv[])
{
    
    const int Blocks_x_ = 32;
    const int Blocks_y_ = 16;

    using real = SCALAR_TYPE;
    using complex = thrust::complex<real>;
    using gpu_vector_operations_real_t = gpu_vector_operations<real>;
    using gpu_vector_operations_complex_t = gpu_vector_operations<complex>;
    using gpu_vector_operations_t = gpu_vector_operations<real>;
    using cufft_type = cufft_wrap_R2C<real>;
    using KF_3D_t = nonlinear_operators::Kolmogorov_3D<cufft_type, 
            gpu_vector_operations_real_t, 
            gpu_vector_operations_complex_t, 
            gpu_vector_operations_t,
            Blocks_x_, Blocks_y_>;     

    using linear_operator_K_3D_stiff_t = nonlinear_operators::linear_operator_K_3D_stiff<gpu_vector_operations_t, KF_3D_t>;
    using preconditioner_K_3D_stiff_t = nonlinear_operators::preconditioner_K_3D_stiff<gpu_vector_operations_t, KF_3D_t, linear_operator_K_3D_stiff_t>;

    using lin_op_t = linear_operator_K_3D_stiff_t;
    using prec_t = preconditioner_K_3D_stiff_t;

    using real_vec_t = typename gpu_vector_operations_real_t::vector_type; 
    using complex_vec_t = typename gpu_vector_operations_complex_t::vector_type;
    using vec_t = typename gpu_vector_operations_t::vector_type;

    using vec_file_ops_t = gpu_file_operations<gpu_vector_operations_t>;


    using log_t = scfd::utils::log_std;
    using monitor_t = numerical_algos::lin_solvers::default_monitor<gpu_vector_operations_t,log_t>;

    using lin_solver_t = numerical_algos::lin_solvers::bicgstabl<lin_op_t,prec_t,gpu_vector_operations_t,monitor_t,log_t>;
    using exact_solver_t = numerical_algos::lin_solvers::exact_wrapper<lin_op_t,prec_t,gpu_vector_operations_t,monitor_t,log_t>;


    size_t N = 64;
    real alpha = 1;
    int one_over_alpha = static_cast<int>(1.0/alpha);
    size_t Nx = N*one_over_alpha;
    size_t Ny = N;
    size_t Nz = N;
    real R = 10.0;
    int gpu_pci_id = 4;
    
    std::cout << "parameters: " << "\nN = " << N << "\none_over_alpha = " << one_over_alpha << "\nNx = " << Nx << " Ny = " << Ny << " Nz = " << Nz << "\nR = " << R << "\ngpu_pci_id = " << gpu_pci_id << "\n";

 
    scfd::utils::init_cuda(gpu_pci_id);

    cufft_type cufft_c2r(Nx, Ny, Nz);
    size_t Mz = cufft_c2r.get_reduced_size();
    cublas_wrap cublas(true);
    cublas.set_pointer_location_device(false);

    log_t log;
    
    
    gpu_vector_operations_real_t vec_ops_r(Nx*Ny*Nz, &cublas);
    gpu_vector_operations_complex_t vec_ops_c(Nx*Ny*Mz, &cublas);
    size_t N_global = 3*(Nx*Ny*Mz-1);
    gpu_vector_operations_t vec_ops(N_global, &cublas);
    
    vec_file_ops_t file_ops(&vec_ops);

    vec_t x0, b0, r0;

    vec_ops.init_vectors(x0, b0, r0); vec_ops.start_use_vectors(x0, b0, r0);

    KF_3D_t kf3d_y(alpha, Nx, Ny, Nz, &vec_ops_r, &vec_ops_c, &vec_ops, &cufft_c2r);
    kf3d_y.randomize_vector(x0);
    // kf3d_y.F(x0, R, b0);
    kf3d_y.randomize_vector(b0);
    kf3d_y.set_linearization_point(b0, R);

    if(!vec_ops.is_valid_number(x0))
    {
        std::cout << "x0 is invalid" << std::endl;
    }
    if(!vec_ops.is_valid_number(b0))
    {
        std::cout << "b0 is invalid" << std::endl;
    }

    //linsolver control
    unsigned int lin_solver_max_it = 1000;
    real lin_solver_tol = 5.0e-9;
    unsigned int use_precond_resid = 1;
    unsigned int resid_recalc_freq = 1;
    unsigned int basis_sz = 3;


    lin_op_t lin_op(&vec_ops, &kf3d_y);
    prec_t prec(&vec_ops, &kf3d_y);
    
    lin_op.set_aE_plus_bA({1, -0.5});

    exact_solver_t exact_solver(&vec_ops, &log);
    exact_solver.set_preconditioner(&prec);


    lin_solver_t lin_solver(&vec_ops, &log); 
    exact_solver.set_preconditioner(&prec);

    lin_solver.set_preconditioner(&prec);
    lin_solver.set_use_precond_resid(use_precond_resid);
    lin_solver.set_resid_recalc_freq(resid_recalc_freq);
    lin_solver.set_basis_size(basis_sz);
    


    auto mon = &exact_solver.monitor();
    mon->init(lin_solver_tol, real(0.0), lin_solver_max_it);
    mon->set_save_convergence_history(true);
    mon->set_divide_out_norms_by_rel_base(true); 

    auto mon1 = &lin_solver.monitor();
    mon1->init(lin_solver_tol, real(0.0), lin_solver_max_it);
    mon1->set_save_convergence_history(true);
    mon1->set_divide_out_norms_by_rel_base(true); 

    vec_ops.assign_scalar(0, x0);
    exact_solver.solve(lin_op, b0, x0);

    // vec_ops.assign_scalar(0, x0);
    // lin_solver.solve(lin_op, b0, x0);

    lin_op.apply(x0, r0);
    vec_ops.add_mul(-1,b0,1,r0);
    std::cout << "resid_norm = " << vec_ops.norm_l2(r0) << ", solution norm = " << vec_ops.norm_l2(x0) << std::endl;

    vec_ops.stop_use_vectors(x0, b0, r0); vec_ops.free_vectors(x0, b0, r0);

    return 0;
}