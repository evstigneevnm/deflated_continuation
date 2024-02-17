#include <cmath>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sstream>
#include <array>
#include <cmath>
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
struct rossler //https://en.wikipedia.org/wiki/R%C3%B6ssler_attractor
{
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;

    rossler(unsigned int used_param_number, T a_init, T b_init, T c_init):
    used_param_number_(used_param_number), param{a_init, b_init, c_init}
    {}
    ~rossler() = default;

    void F(const T time_p, const T_vec& in_p, const T param_p, T_vec& out_p )const
    {
        param[used_param_number_] = param_p;

        out_p[0] = -in_p[1]-in_p[2];
        out_p[1] = in_p[0]+param[0]*in_p[1];
        out_p[2] = param[1] + in_p[2]*(in_p[0]-param[2]);
    }

    void set_initial(T_vec& x0)const
    {
        x0[0] = 2.0;
        x0[1] = 0.0;
        x0[2] = 0.0;
    }

    void norm_bifurcation_diagram(const T_vec& x0, std::vector<T>& norm_vec)const
    {
        norm_vec.push_back(x0[0]);
        norm_vec.push_back(x0[1]);
        norm_vec.push_back(x0[2]);
    }
    T check_solution_quality(const T_vec& x)const
    {
        bool finite = true;
        for(int j = 0;j<3;j++)
        {
            finite &= std::isfinite(x[j]);
        }
        return finite;
    }

    T get_selected_parameter_value()const
    {
        return param[used_param_number_];
    }

private:
    unsigned int used_param_number_;
    mutable std::array<T, 3> param;
    // some parameters:
    // a = 0.2 b = 0.2 c = 5.7
    // a = 0.2 b = 0.2 c = 14.0
    // standard bifurcaitons:
    // a=0.1, b=0.1:
    // c = 4, period-1 orbit.
    // c = 6, period-2 orbit.
    // c = 8.5, period-4 orbit.
    // c = 8.7, period-8 orbit.
    // c = 9, sparse chaotic attractor.
    // c = 12, period-3 orbit.
    // c = 12.6, period-6 orbit.
    // c = 13, sparse chaotic attractor.
    // c = 18, filled-in chaotic attractor.

};
}



int main(int argc, char const *argv[])
{

    using real = SCALAR_TYPE;
    using log_t = utils::log_std;

    using vec_ops_t = cpu_vector_operations<real>;
    using vec_t = typename vec_ops_t::vector_type;

    using nlin_op_t = nonlinear_operators::rossler<vec_ops_t>;
    
    using time_step_const_t = time_steppers::time_step_adaptation_constant<vec_ops_t, log_t>;
    using time_step_err_ctrl_t = time_steppers::time_step_adaptation_error_control<vec_ops_t, log_t>;
    
    using single_step_const_t = time_steppers::explicit_time_step<vec_ops_t, nlin_op_t, log_t, time_step_const_t>;
    using single_step_err_ctrl_t = time_steppers::explicit_time_step<vec_ops_t, nlin_op_t, log_t, time_step_err_ctrl_t>;
    
    using time_stepper_const_t = time_steppers::time_stepper<vec_ops_t, nlin_op_t, single_step_const_t,log_t>;
    using time_stepper_err_ctrl_t = time_steppers::time_stepper<vec_ops_t, nlin_op_t, single_step_err_ctrl_t,log_t>;


    if(argc != 7)
    {
        std::cout << argv[0] << " a b c select time name\n  a,b,c - parameters,\n select - main parameter number (0, 1 or 2), \n time - simulation time,\n";
        std::cout << "   name - name of the scheme: EE, HE, RK33SSP, RK43SSP, RKDP45, RK64SSP" << std::endl;
        return(0);       
    }    
    real a = std::stof(argv[1]);
    real b = std::stof(argv[2]);
    real c = std::stof(argv[3]);
    unsigned int select = std::stoi(argv[4]);
    real simulation_time = std::stof(argv[5]);
    std::string scheme_name(argv[6]);

    log_t log;

    vec_ops_t vec_ops(3);
    
    vec_t x0;

    vec_ops.init_vector(x0); vec_ops.start_use_vector(x0);


    nlin_op_t rossler(select, a, b, c); //use second parameter as a bifurcation parameter.
    rossler.set_initial(x0);

    
    time_step_const_t time_step_const(&vec_ops, &log);
    time_step_err_ctrl_t time_step_err_ctrl(&vec_ops, &log);

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
    auto mu = rossler.get_selected_parameter_value();

    single_step_const_t explicit_step_const(&vec_ops, &time_step_const, &log, &rossler, mu, method);
    single_step_err_ctrl_t explicit_step_err_control(&vec_ops, &time_step_err_ctrl, &log, &rossler, mu, method);
    
    time_stepper_const_t time_stepper_const(&vec_ops, &rossler, &explicit_step_const, &log);
    time_stepper_err_ctrl_t time_stepper_err_ctrl(&vec_ops, &rossler, &explicit_step_err_control, &log);

    log.info_f("executing time stepper with time = %.2le", simulation_time);
    // time_stepper_err_ctrl.set_parameter(mu);
    // time_stepper_err_ctrl.set_initial_conditions(x0, 0.0);
    // time_stepper_err_ctrl.execute();

    time_stepper_err_ctrl.execute(x0, mu, {0, simulation_time} );
    std::stringstream ss;
    ss << "rossler_result_" << scheme_name << ".dat";
    time_stepper_err_ctrl.save_norms( ss.str() );
    // std::stringstream ss;
    // ss << "x_" << simulation_time << "_sim.pos";


    vec_ops.stop_use_vector(x0); vec_ops.free_vector(x0);
    return 0;
}

