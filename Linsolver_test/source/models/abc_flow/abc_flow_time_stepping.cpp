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

#include <utils/log.h>
#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/bicgstabl.h>

#include <nonlinear_operators/abc_flow/abc_flow.h>
#include <nonlinear_operators/abc_flow/linear_operator_abc_flow.h>
#include <nonlinear_operators/abc_flow/preconditioner_abc_flow.h>
#include <nonlinear_operators/abc_flow/system_operator.h>
#include <nonlinear_operators/abc_flow/convergence_strategy.h>

#include <numerical_algos/newton_solvers/newton_solver.h>

#include <common/macros.h>
#include <common/gpu_file_operations.h>
#include <common/gpu_vector_operations.h>

#include <time_stepper/time_step_adaptation_error_control.h>
#include <time_stepper/explicit_time_step.h>
#include <time_stepper/time_stepper.h>
#include <time_stepper/distance_to_points.h>


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
    using abc_flow_t = nonlinear_operators::abc_flow<cufft_type, 
            gpu_vector_operations_real_t, 
            gpu_vector_operations_complex_t, 
            gpu_vector_operations_t,
            Blocks_x_, Blocks_y_>;
    using real_vec_t = typename gpu_vector_operations_real_t::vector_type; 
    using complex_vec_t = typename gpu_vector_operations_complex_t::vector_type;
    using vec_t = typename gpu_vector_operations_t::vector_type;

    using vec_file_ops_t = gpu_file_operations<gpu_vector_operations_t>;


    using log_t = utils::log_std;
    using monitor_t = numerical_algos::lin_solvers::default_monitor<gpu_vector_operations_t,log_t>;
    using time_step_err_ctrl_t = time_steppers::time_step_adaptation_error_control<gpu_vector_operations_t, log_t>;
    using time_step_t = time_steppers::explicit_time_step<gpu_vector_operations_t, abc_flow_t, log_t, time_step_err_ctrl_t>;
    using distance_to_points_t = time_steppers::distance_to_points<gpu_vector_operations_t, abc_flow_t>;
    using time_stepper_t = time_steppers::time_stepper<gpu_vector_operations_t, abc_flow_t, time_step_t, log_t, distance_to_points_t>;

    if((argc < 5)||(argc > 7))
    {
        std::cout << argv[0] << " N R time method [state_file skip_solution_plot_time](optional)\n  R -- Reynolds number,\n  N = 2^n- discretization in one direction. \n  time - simmulation time. \n   skip_solution_plot_time - to save solution files at given time intervals. \n";
         std::cout << " method - name of the scheme: EE, HE, RK33SSP, RK43SSP, RKDP45, RK64SSP" << std::endl;
        return(0);       
    }    
    size_t N = std::stoul(argv[1]);
    size_t Nx = N;
    size_t Ny = N;
    size_t Nz = N;
    real R = std::stof(argv[2]);
    real simulation_time = std::stof(argv[3]);
    std::string scheme_name(argv[4]);
    bool load_file = false;
    std::string load_file_name;
    if((argc == 6)||(argc == 7))
    {
        load_file = true;
        load_file_name = std::string(argv[5]);
    }
    real skip_solution_plot_time = 0;
    if(argc == 7)
    {
        skip_solution_plot_time = std::stof(argv[6]);
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
    int gpu_pci_id = 28;
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
    gpu_vector_operations_t vec_ops(6*(Nx*Ny*Mz-1), &cublas);
    
    vec_file_ops_t file_ops(&vec_ops);

    vec_t x0;

    vec_ops.init_vector(x0); vec_ops.start_use_vector(x0);

    abc_flow_t abc_flow(Nx, Ny, Nz, &vec_ops_r, &vec_ops_c, &vec_ops, &cufft_c2r);

    distance_to_points_t distance_to_points(&vec_ops, &abc_flow);
    distance_to_points.set_drop_solutions(skip_solution_plot_time, "t_abc_abs.pos");

    if(load_file)
    {
        file_ops.read_vector(load_file_name, x0);
    }
    else
    {
        abc_flow.randomize_vector(x0);
    }

    time_step_err_ctrl_t time_step_err_ctrl(&vec_ops, &log, {0.0, simulation_time});

    time_step_t explicit_step(&vec_ops, &time_step_err_ctrl,  &log, &abc_flow, R, scheme_name);//explicit_step(&vec_ops, &abc_flow, &log);
    time_stepper_t time_stepper(&vec_ops, &abc_flow, &explicit_step, &log, &distance_to_points);
    
    log.info_f("executing time stepper with time = %.2le", simulation_time);

    time_stepper.set_parameter(R);
    // time_stepper.set_time(simulation_time);
    time_stepper.set_initial_conditions(x0);
    // time_stepper.get_results(x0);
    time_stepper.set_skip(1000);
    time_stepper.execute();
    time_stepper.save_norms("probe_1.dat");
    time_stepper.get_results(x0);



    std::stringstream ss, ss1;
    ss << "x_" << simulation_time << "_sim.pos";
    auto pos_file_name(ss.str());
    abc_flow.write_solution_abs(pos_file_name, x0);     

    ss1 << "x_" << simulation_time << "_R_" << R << "_res.dat";
    file_ops.write_vector(ss1.str(), x0);


    vec_ops.stop_use_vector(x0); vec_ops.free_vector(x0);
    return 0;
}

