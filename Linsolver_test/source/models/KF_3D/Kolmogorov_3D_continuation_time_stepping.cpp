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

#include <nonlinear_operators/Kolmogorov_flow_3D/Kolmogorov_3D.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/linear_operator_K_3D.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/preconditioner_K_3D.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/system_operator.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/convergence_strategy.h>

#include <numerical_algos/newton_solvers/newton_solver.h>

#include <common/macros.h>
#include <common/gpu_file_operations.h>
#include <common/gpu_vector_operations.h>

#include <time_stepper/time_step_adaptation_error_control.h>
#include <time_stepper/time_step_adaptation_constant.h>
#include <time_stepper/explicit_time_step.h>
#include <time_stepper/time_stepper.h>


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
    using real_vec_t = typename gpu_vector_operations_real_t::vector_type; 
    using complex_vec_t = typename gpu_vector_operations_complex_t::vector_type;
    using vec_t = typename gpu_vector_operations_t::vector_type;

    using vec_file_ops_t = gpu_file_operations<gpu_vector_operations_t>;


    using log_t = scfd::utils::log_std;
    using monitor_t = numerical_algos::lin_solvers::default_monitor<gpu_vector_operations_t,log_t>;
    using time_step_err_ctrl_t = time_steppers::time_step_adaptation_error_control<gpu_vector_operations_t, log_t>;
    using time_step_adaptation_constant_t = time_steppers::time_step_adaptation_constant<gpu_vector_operations_t, log_t>;

    using time_step_t = time_steppers::explicit_time_step<gpu_vector_operations_t, KF_3D_t, log_t, time_step_err_ctrl_t>;
    using time_stepper_t = time_steppers::time_stepper<gpu_vector_operations_t, KF_3D_t, time_step_t, log_t>;

    if((argc < 7)||(argc > 8))
    {
        std::cout << argv[0] << " N alpha R time method GPU_PCI_ID state_file(optional)\n   0<alpha<=1 - torus stretching parameter,\n R -- Reynolds number,\n  N = 2^n- discretization in one direction. \n  time - simulation time. \n";
         std::cout << " method - name of the scheme: EE, HE, RK33SSP, RK43SSP, RKDP45, RK64SSP" << std::endl;
        return(0);       
    }    
    real initial_dt = 5.0e-3;
    size_t N = std::stoul(argv[1]);
    real alpha = std::stof(argv[2]);
    int one_over_alpha = static_cast<int>(1.0/alpha);
    size_t Nx = N*one_over_alpha;
    size_t Ny = N;
    size_t Nz = N;
    real R = std::stof(argv[3]);
    real simulation_time = std::stof(argv[4]);
    std::string scheme_name(argv[5]);
    int gpu_pci_id = std::stoi(argv[6]);
    bool load_file = false;
    std::string load_file_name;
    if(argc == 8)
    {
        load_file = true;
        load_file_name = std::string(argv[7]);
    }

    std::cout << "input parameters: " << "\nN = " << N << "\none_over_alpha = " << one_over_alpha << "\nNx = " << Nx << " Ny = " << Ny << " Nz = " << Nz << "\nR = " << R << "\nsimulation_time = " << simulation_time << "\nscheme_name = " << scheme_name << "\ngpu_pci_id = " << gpu_pci_id << "\n";
//  sqrt(L*scale_force)/(n R):
    const real scale_force = 0.1;
    std::cout << "reduced Reynolds number = " <<  std::sqrt(one_over_alpha*2*3.141592653589793238*scale_force)*R << std::endl;
    if(load_file)
    {
        std::cout << "\nload_file_name = " << load_file_name;
    }
    std::cout  << std::endl;

    auto method = time_steppers::detail::methods::EXPLICIT_EULER;
    if(scheme_name == "EE")
    {
        method = time_steppers::detail::methods::EXPLICIT_EULER;
    }
    else if(scheme_name == "RKDP45")
    {
        method = time_steppers::detail::methods::RKDP45;
    }
    else if(scheme_name == "RK33SSP")
    {
        method = time_steppers::detail::methods::RK33SSP;
    }    
    else if(scheme_name == "RK43SSP")
    {
        method = time_steppers::detail::methods::RK43SSP;
    } 
    else if(scheme_name == "RK64SSP")
    {
        method = time_steppers::detail::methods::RK64SSP;
    }     
    else if(scheme_name == "HE")
    {
        method = time_steppers::detail::methods::HEUN_EULER;
    }  
    else
    {
        throw std::logic_error("incorrect method string type provided.");
    }

    scfd::utils::init_cuda(gpu_pci_id);

    cufft_type cufft_c2r(Nx, Ny, Nz);
    size_t Mz = cufft_c2r.get_reduced_size();
    cublas_wrap cublas(true);
    cublas.set_pointer_location_device(false);

    log_t log;
    log_t log_ls;
    log_ls.set_verbosity(0);

    
    gpu_vector_operations_real_t vec_ops_r(Nx*Ny*Nz, &cublas);
    gpu_vector_operations_complex_t vec_ops_c(Nx*Ny*Mz, &cublas);
    size_t N_global = 3*(Nx*Ny*Mz-1);
    gpu_vector_operations_t vec_ops(N_global, &cublas);
    
    vec_file_ops_t file_ops(&vec_ops);

    vec_t x0;

    vec_ops.init_vector(x0); vec_ops.start_use_vector(x0);

    KF_3D_t kf3d_y(alpha, Nx, Ny, Nz, &vec_ops_r, &vec_ops_c, &vec_ops, &cufft_c2r);

    if(load_file)
    {
        file_ops.read_vector(load_file_name, x0);
    }
    else
    {
        kf3d_y.randomize_vector(x0);
    }

    time_step_err_ctrl_t time_step_err_ctrl(&vec_ops, &log, {0.0, simulation_time});
    time_step_adaptation_constant_t time_step_const(&vec_ops, &log, {0.0, simulation_time}, initial_dt);

    time_step_t explicit_step(&vec_ops, &time_step_err_ctrl, &log, &kf3d_y, R, method);//explicit_step(&vec_ops, &kf3d_y, &log);
    time_stepper_t time_stepper(&vec_ops, &kf3d_y, &explicit_step, &log);
    
    log.info_f("executing time stepper with time = %.2le", simulation_time);

    time_stepper.set_parameter(R);
    // time_stepper.set_time(simulation_time);
    time_stepper.set_initial_conditions(x0);
    // time_stepper.get_results(x0);
    // time_stepper.set_skip(1000);
    time_stepper.execute();
    time_stepper.save_norms("probe_1.dat");
    time_stepper.get_results(x0);



    std::stringstream ss, ss1;
    ss << "x_" << simulation_time << "_sim.pos";
    auto pos_file_name(ss.str());
    kf3d_y.write_solution_abs(pos_file_name, x0);     

    ss1 << "x_" << simulation_time << "_R_" << R << "_res.dat";
    file_ops.write_vector(ss1.str(), x0);


    vec_ops.stop_use_vector(x0); vec_ops.free_vector(x0);
    return 0;
}

