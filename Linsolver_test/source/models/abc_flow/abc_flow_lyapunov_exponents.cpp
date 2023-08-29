#include <cmath>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sstream>
#include <fstream>
#include <thrust/complex.h>
#include <utils/cuda_support.h>

#include <utils/init_cuda.h>
#include <external_libraries/cufft_wrap.h>
#include <external_libraries/cublas_wrap.h>
#include <external_libraries/lapack_wrap.h>

#include <utils/log.h>
#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/bicgstabl.h>

#include <nonlinear_operators/abc_flow/abc_flow.h>

#include <time_stepper/time_step_adaptation_constant.h>
#include <time_stepper/time_step_adaptation_error_control.h>
#include <time_stepper/explicit_time_step.h>
#include <time_stepper/lyapunov_exponents.h>

#include <common/gpu_file_operations.h>
#include <common/gpu_vector_operations.h>
#include <common/gpu_matrix_vector_operations.h>
#include <stability/IRAM/iram_process.hpp>


int main(int argc, char const *argv[])
{
    const int Blocks_x_ = 64;
    const int Blocks_y_ = 16;
    
    using real = SCALAR_TYPE;
    using complex = thrust::complex<real>;
    using gpu_vector_operations_real_t = gpu_vector_operations<real>;
    using gpu_vector_operations_complex_t = gpu_vector_operations<complex>;
    using vec_ops_t = gpu_vector_operations<real>;
    using vec_t = typename vec_ops_t::vector_type;
    using mat_ops_t = gpu_matrix_vector_operations<real, vec_t>;
    using mat_t = typename mat_ops_t::matrix_type;

    using vec_file_ops_t = gpu_file_operations<vec_ops_t>;

    using cufft_type = cufft_wrap_R2C<real>;
    using abc_flow_t = nonlinear_operators::abc_flow<cufft_type, 
            gpu_vector_operations_real_t, 
            gpu_vector_operations_complex_t, 
            vec_ops_t,
            Blocks_x_, Blocks_y_>;    


    using log_t = utils::log_std;
    using monitor_t = numerical_algos::lin_solvers::default_monitor<vec_ops_t,log_t>;

    using lyapunov_exp_t = time_steppers::lyapunov_exponents<vec_ops_t, abc_flow_t, time_steppers::time_step_adaptation_error_control, time_steppers::explicit_time_step, log_t, 6>;



    if((argc < 6)||(argc > 8))
    {
        std::cout << argv[0] << " cuda_dev_num N R time iterations method state_file(optional)\n  R -- Reynolds number,\n  N = 2^n- discretization in one direction. \n  time - simmulation time. \n";
         std::cout << " method - name of the scheme: EE, HE, RK33SSP, RK43SSP, RKDP45, RK64SSP" << std::endl;
        return(0);       
    }    
    int cuda_dev_num = std::stoi(argv[1]);
    size_t N = std::stoul(argv[2]);
    size_t Nx = N;
    size_t Ny = N;
    size_t Nz = N;
    real R = std::stof(argv[3]);
    real simulation_time = std::stof(argv[4]);
    size_t iterations = std::stoul(argv[5]);
    std::string scheme_name(argv[6]);
    bool load_file = false;
    std::string load_file_name;
        
    if(argc == 8)
    {
        load_file = true;
        load_file_name = std::string(argv[7]);
    }

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

    if( !utils::init_cuda(cuda_dev_num) )
    {
        return 2;
    } 
    cufft_type cufft_c2r(Nx, Ny, Nz);
    size_t Mz = cufft_c2r.get_reduced_size();
    cublas_wrap cublas(true);
    cublas.set_pointer_location_device(false);

    log_t log;
    log_t log_ls;
    log_ls.set_verbosity(0);
    log.info_f("Executing Lyapunov exponents for R = %f with total iterations = %i, each of time = %f", R, iterations, simulation_time);
    
    gpu_vector_operations_real_t vec_ops_r(Nx*Ny*Nz, &cublas);
    gpu_vector_operations_complex_t vec_ops_c(Nx*Ny*Mz, &cublas);
    vec_ops_t vec_ops(6*(Nx*Ny*Mz-1), &cublas);
    vec_file_ops_t file_ops(&vec_ops);

    vec_t x0,x1;

    vec_ops.init_vector(x0); vec_ops.start_use_vector(x0);
    vec_ops.init_vector(x1); vec_ops.start_use_vector(x1);

    abc_flow_t abc_flow(Nx, Ny, Nz, &vec_ops_r, &vec_ops_c, &vec_ops, &cufft_c2r);

    if(load_file)
    {
        file_ops.read_vector(load_file_name, x0);
    }
    else
    {
        abc_flow.randomize_vector(x0);
    }


    lyapunov_exp_t lyapunov_exp(&vec_ops, &abc_flow, &log, simulation_time, R, method);


    // lyapunov_exp.run_single_time(x0, x1);
    lyapunov_exp.apply(iterations, x0, x1);

    std::stringstream ss_t;
    ss_t << "x_lyapunov_" << simulation_time << "_R_" << R << ".dat";
    file_ops.write_vector(ss_t.str(), x1);

    std::stringstream ss_e;
    ss_e << "x_lyapunov_exponents_" << iterations << "_R_" << R << ".dat";
    lyapunov_exp.save_exponents(ss_e.str() );
    


    vec_ops.stop_use_vector(x1); vec_ops.free_vector(x1);
    vec_ops.stop_use_vector(x0); vec_ops.free_vector(x0);

    return 0;
}