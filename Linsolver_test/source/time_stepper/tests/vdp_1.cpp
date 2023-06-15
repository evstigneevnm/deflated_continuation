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

    void F(const T time_p, const T_vec& in_p, const T param_p, T_vec& out_p ) const
    {
        // dydt = [y(2); mu*(1-y(1)^2)*y(2)-y(1)];
        out_p[0] = in_p[1];
        out_p[1] = param_p*(1-in_p[0])*(1-in_p[0])*in_p[1]-in_p[0];
    }

    void set_initial(T_vec& x0)const
    {
        x0[0] = 2.0;
        x0[1] = 0.0;
    }

    void norm_bifurcation_diagram(const T_vec& x0, std::vector<T>& norm_vec)
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
    
    using time_step_t = time_steppers::time_step_adaptation_constant<vec_ops_t, log_t>;
    using single_step_t = time_steppers::explicit_time_step<vec_ops_t, nlin_op_t, log_t, time_step_t>;
    using time_stepper_t = time_steppers::time_stepper<vec_ops_t, nlin_op_t, single_step_t,log_t>;

    if(argc != 3)
    {
        std::cout << argv[0] << " mu time\n  mu - parameter, time - simulation time.";
        return(0);       
    }    
    real mu = std::stof(argv[1]);
    real simulation_time = std::stof(argv[2]);
    
    log_t log;

    vec_ops_t vec_ops(2);
    
    vec_t x0;

    vec_ops.init_vector(x0); vec_ops.start_use_vector(x0);


    nlin_op_t vdp;
    vdp.set_initial(x0);

    
    time_step_t time_step(&vec_ops, &log, {0.0, simulation_time}, 1.0e-4);
    single_step_t explicit_step(&vec_ops, &vdp, &time_step, &log, mu, time_steppers::detail::methods::EXPLICIT_EULER);
    time_stepper_t time_stepper(&vec_ops, &vdp, &explicit_step, &log);
    

    log.info_f("executing time stepper with time = %.2le", simulation_time);
    time_stepper.set_parameter(mu);
    time_stepper.set_initial_conditions(x0, 0.0);
    time_stepper.execute();

    // std::stringstream ss;
    // ss << "x_" << simulation_time << "_sim.pos";
  

    vec_ops.stop_use_vector(x0); vec_ops.free_vector(x0);
    return 0;
}

