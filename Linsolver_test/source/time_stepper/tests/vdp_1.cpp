#include <cmath>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sstream>

// #include <utils/init_cuda.h>

#include <utils/log.h>
// #include <numerical_algos/lin_solvers/default_monitor.h>
// #include <numerical_algos/lin_solvers/bicgstabl.h>

// #include <common/macros.h>
// #include <common/gpu_file_operations.h>
// #include <common/gpu_vector_operations.h>
#include <common/file_operations.h>
#include <common/cpu_vector_operations.h>

#include <time_stepper/time_step_adaptation_constant.h>
#include <time_stepper/time_step_adaptation_error_control.h>
#include <time_stepper/explicit_time_step.h>
#include <time_stepper/time_stepper.h>


namespace nonlinear_operators
{

template<class VectorOperations>
struct vdp
{
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;

    vdp() = default;
    ~vdp() = default;

    void F(const T time_p, const T_vec& in_p, const T param_p, T_vec& out_p )const
    {
        // dydt = [y(2); mu*(1-y(1)^2)*y(2)-y(1)];
        out_p[0] = in_p[1];
        out_p[1] = param_p*(1-in_p[0]*in_p[0])*in_p[1]-in_p[0];
    }

    void set_initial(T_vec& x0)const
    {
        x0[0] = 2.0;
        x0[1] = 0.0;
    }

    void norm_bifurcation_diagram(const T_vec& x0, std::vector<T>& norm_vec)const
    {
        norm_vec.push_back(x0[0]);
        norm_vec.push_back(x0[1]);
    }


};
}



int main(int argc, char const *argv[])
{

    using real = SCALAR_TYPE;
    using log_t = utils::log_std;

    using vec_ops_t = cpu_vector_operations<real>;
    using vec_t = typename vec_ops_t::vector_type;

    using nlin_op_t = nonlinear_operators::vdp<vec_ops_t>;
    
    using time_step_const_t = time_steppers::time_step_adaptation_constant<vec_ops_t, log_t>;
    using time_step_err_ctrl_t = time_steppers::time_step_adaptation_error_control<vec_ops_t, log_t>;
    
    using single_step_const_t = time_steppers::explicit_time_step<vec_ops_t, nlin_op_t, log_t, time_step_const_t>;
    using single_step_err_ctrl_t = time_steppers::explicit_time_step<vec_ops_t, nlin_op_t, log_t, time_step_err_ctrl_t>;
    
    using time_stepper_const_t = time_steppers::time_stepper<vec_ops_t, nlin_op_t, single_step_const_t,log_t>;
    using time_stepper_err_ctrl_t = time_steppers::time_stepper<vec_ops_t, nlin_op_t, single_step_err_ctrl_t,log_t>;


    if(argc != 4)
    {
        std::cout << argv[0] << " mu time name\n  mu - parameter, time - simulation time,\n";
        std::cout << "   name - name of the scheme: EE, HE, RK33SSP, RK43SSP, RKDP45, RK64SSP" << std::endl;
        return(0);       
    }    
    real mu = std::stof(argv[1]);
    real simulation_time = std::stof(argv[2]);
    std::string scheme_name(argv[3]);

    log_t log;

    vec_ops_t vec_ops(2);
    
    vec_t x0;

    vec_ops.init_vector(x0); vec_ops.start_use_vector(x0);


    nlin_op_t vdp;
    vdp.set_initial(x0);

    
    time_step_const_t time_step_const(&vec_ops, &log, {0.0, simulation_time}, 5.0e-3);
    time_step_err_ctrl_t time_step_err_ctrl(&vec_ops, &log, {0.0, simulation_time});

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


    single_step_const_t explicit_step_const(&vec_ops, &vdp, &time_step_const, &log, mu, method);
    single_step_err_ctrl_t explicit_step_err_control(&vec_ops, &vdp, &time_step_err_ctrl, &log, mu, method);
    
    time_stepper_const_t time_stepper_const(&vec_ops, &vdp, &explicit_step_const, &log);
    time_stepper_err_ctrl_t time_stepper_err_ctrl(&vec_ops, &vdp, &explicit_step_err_control, &log);

    log.info_f("executing time stepper with time = %.2le", simulation_time);
    time_stepper_err_ctrl.set_parameter(mu);
    time_stepper_err_ctrl.set_initial_conditions(x0, 0.0);
    time_stepper_err_ctrl.execute();
    std::stringstream ss;
    ss << "vdp_result_" << scheme_name << ".dat";
    time_stepper_err_ctrl.save_norms( ss.str() );
    // std::stringstream ss;
    // ss << "x_" << simulation_time << "_sim.pos";
  

    vec_ops.stop_use_vector(x0); vec_ops.free_vector(x0);
    return 0;
}

