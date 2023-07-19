#include <cmath>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sstream>
#include <thrust/complex.h>
#include <utils/cuda_support.h>

#include <utils/init_cuda.h>
#include <external_libraries/cufft_wrap.h>
#include <external_libraries/cublas_wrap.h>

#include <utils/log.h>
#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/bicgstabl.h>

#include <nonlinear_operators/abc_flow/abc_flow.h>

#include <time_stepper/time_step_adaptation_constant.h>
#include <time_stepper/time_step_adaptation_error_control.h>
#include <time_stepper/explicit_time_step.h>
#include <periodic_orbit/periodic_orbit_nonlinear_operator.h>
#include <periodic_orbit/system_operator_single_section.h>
#include <periodic_orbit/convergence_strategy_single_section.h>
#include <numerical_algos/newton_solvers/newton_solver.h>

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
    using vec_ops_t = gpu_vector_operations<real>;
    using vec_t = typename vec_ops_t::vector_type;
    
    using vec_file_ops_t = gpu_file_operations<vec_ops_t>;

    using cufft_type = cufft_wrap_R2C<real>;
    using abc_flow_t = nonlinear_operators::abc_flow<cufft_type, 
            gpu_vector_operations_real_t, 
            gpu_vector_operations_complex_t, 
            vec_ops_t,
            Blocks_x_, Blocks_y_>;    


    using log_t = utils::log_std;
    using monitor_t = numerical_algos::lin_solvers::default_monitor<vec_ops_t,log_t>;

    using periodic_orbit_nonlinear_operator_t = periodic_orbit::periodic_orbit_nonlinear_operator<vec_ops_t, abc_flow_t, log_t, time_steppers::time_step_adaptation_error_control, time_steppers::explicit_time_step>;

    using periodic_orbit_linear_operator_t = typename periodic_orbit_nonlinear_operator_t::linear_operator_type;
    using periodic_orbit_preconditioner_t = typename periodic_orbit_nonlinear_operator_t::preconditioner_type;

    using lin_solve_t = numerical_algos::lin_solvers::bicgstabl<periodic_orbit_linear_operator_t, periodic_orbit_preconditioner_t, vec_ops_t, monitor_t, log_t>;

    using system_operator_single_section_t = nonlinear_operators::newton_method::system_operator_single_section<vec_ops_t, periodic_orbit_nonlinear_operator_t, periodic_orbit_linear_operator_t, lin_solve_t>;    
    using convergence_strategy_single_section_t = nonlinear_operators::newton_method::convergence_strategy_single_section<vec_ops_t, periodic_orbit_nonlinear_operator_t, log_t>;
    using newton_solver_t = numerical_algos::newton_method::newton_solver<vec_ops_t, periodic_orbit_nonlinear_operator_t, system_operator_single_section_t, convergence_strategy_single_section_t>;

    if((argc < 5)||(argc > 6))
    {
        std::cout << argv[0] << " N R time method state_file(optional)\n  R -- Reynolds number,\n  N = 2^n- discretization in one direction. \n  time - simmulation time. \n";
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
    if(argc == 6)
    {
        load_file = true;
        load_file_name = std::string(argv[5]);
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

    if( !utils::init_cuda() )
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

    
    gpu_vector_operations_real_t vec_ops_r(Nx*Ny*Nz, &cublas);
    gpu_vector_operations_complex_t vec_ops_c(Nx*Ny*Mz, &cublas);
    vec_ops_t vec_ops(6*(Nx*Ny*Mz-1), &cublas);
    vec_file_ops_t file_ops(&vec_ops);
    vec_t x0;

    vec_ops.init_vector(x0); vec_ops.start_use_vector(x0);

    abc_flow_t abc_flow(Nx, Ny, Nz, &vec_ops_r, &vec_ops_c, &vec_ops, &cufft_c2r);

    if(load_file)
    {
        file_ops.read_vector(load_file_name, x0);
    }
    else
    {
        abc_flow.randomize_vector(x0);
    }
    periodic_orbit_nonlinear_operator_t periodic_orbit_nonlin_op(&vec_ops, &abc_flow, &log, 100.0, R, method);

    auto periodic_orbit_lin_op = periodic_orbit_nonlin_op.linear_operator;

    lin_solve_t solver(&vec_ops, &log, 0);
    solver.set_basis_size(3);
    real rel_tol = 1.0e-2;
    size_t max_iters = 100;
    auto& mon = solver.monitor();
    mon.init(rel_tol, real(0), max_iters);
    mon.set_save_convergence_history(true);
    mon.set_divide_out_norms_by_rel_base(true);    

    real newton_rol = 1.0e-7;
    system_operator_single_section_t sys_op(&vec_ops, periodic_orbit_lin_op, &solver);
    convergence_strategy_single_section_t convergence(&vec_ops, &log, newton_rol);
    newton_solver_t newton(&vec_ops, &sys_op, &convergence);

    periodic_orbit_nonlin_op.set_hyperplane_from_initial_guesses(x0, R);
    if(simulation_time>0.0)
    {
        periodic_orbit_nonlin_op.time_stepper(x0, R, {0, simulation_time});
        periodic_orbit_nonlin_op.save_norms("abc_initial.dat");
    }

    bool is_newton_converged = newton.solve(&periodic_orbit_nonlin_op, x0, R);

    if(is_newton_converged)
    {
        std::stringstream ss_periodic_estimate;
        ss_periodic_estimate << "abc_period_R_" << R << "_scheme_" << scheme_name << ".dat";
        periodic_orbit_nonlin_op.save_period_estmate_norms(ss_periodic_estimate.str() );

        std::stringstream ss, ss1;
        ss << "x_" << simulation_time << "_R_" << R << "_periodic.pos";
        auto pos_file_name(ss.str());
        abc_flow.write_solution_abs(pos_file_name, x0);     

        ss1 << "x_" << simulation_time << "_R_" << R << "_periodic.dat";
        file_ops.write_vector(ss1.str(), x0);
    }

    vec_ops.stop_use_vector(x0); vec_ops.free_vector(x0);

    return 0;
}