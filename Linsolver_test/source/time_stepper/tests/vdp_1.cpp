#include <cmath>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sstream>

// #include <utils/init_cuda.h>

#include <scfd/utils/log.h>
#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/bicgstab.h>
#include <numerical_algos/lin_solvers/bicgstabl.h>

// #include <common/macros.h>
// #include <common/gpu_file_operations.h>
// #include <common/gpu_vector_operations.h>
#include <common/file_operations.h>
#include <common/cpu_vector_operations.h>

#include <time_stepper/detail/butcher_tables.h>



#include <time_stepper/time_step_adaptation_constant.h>
#include <time_stepper/time_step_adaptation_error_control.h>
#include <time_stepper/time_step_adaptation_tolerance.h>
#include <time_stepper/explicit_time_step.h>
#include <time_stepper/implicit_time_step.h>
#include <time_stepper/time_stepper.h>


namespace nonlinear_operators
{

template<class VectorOperations>
struct vdp
{
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;

    vdp():lin_op(this), prec() {};
    ~vdp() = default;

    void F(const T time_p, const T_vec& in_p, const T param_p, T_vec& out_p )const
    {
        // dy/dt = [y(2); mu*(1-y(1)^2)*y(2)-y(1)];
        out_p[0] = in_p[1];
        out_p[1] = param_p*(1-in_p[0]*in_p[0])*in_p[1]-in_p[0];
    }

    void jacobian(const T_vec& du, T_vec& dv)const
    {
        // 0             1
        // 2mu*xy-1   mu*(1-x^2)
        // std::cout << "du = " << du[0] << " " << du[1] << std::endl;
        dv[0] = du[1];
        dv[1] = (-2.0*param_0*u_0[0]*u_0[1]-1.0)*du[0] + param_0*(1-u_0[0]*u_0[0])*du[1];
        // std::cout << "dv = " << dv[0] << " " << dv[1] << std::endl;
    }    

    void set_linearization_point(const T_vec& u_p_0, const T param_p_0)
    {
        u_0 = u_p_0;
        param_0 = param_p_0;
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
    T check_solution_quality(const T_vec& x)const
    {
        return 0.0;
    }

    struct linear_operator
    {
        linear_operator(vdp* ref_p): encl(ref_p)
        {}

        void set_aE_plus_bA(const std::pair<T,T>& ab_pair)
        {
            ab = ab_pair;
        }  
        void apply(const T_vec& x, T_vec& f)const
        {
            encl->jacobian(x, f);
            //calc: y := mul_x*x + mul_y*y
            // vec_ops->add_mul(ab.first, x, ab.second, f);
            f[0] = ab.first*x[0] + ab.second*f[0];
            f[1] = ab.first*x[1] + ab.second*f[1];
        }        

        std::pair<T, T> ab;
        vdp* encl;
    };

    struct preconditioner_dummy
    {
        preconditioner_dummy() = default;

        void apply(T_vec& x0)
        { }
        void set_operator(const linear_operator *op_)
        { }


    };

    T param_0;
    T_vec u_0;
    linear_operator lin_op;
    preconditioner_dummy prec;

};


}



int main(int argc, char const *argv[])
{

    using real = SCALAR_TYPE;
    using log_t = scfd::utils::log_std;

    using vec_ops_t = cpu_vector_operations<real>;
    using vec_t = typename vec_ops_t::vector_type;
    using monitor_t = numerical_algos::lin_solvers::default_monitor<vec_ops_t, log_t>;

    using nlin_op_t = nonlinear_operators::vdp<vec_ops_t>;
    using linear_op_t = nlin_op_t::linear_operator;
    using precond_t = nlin_op_t::preconditioner_dummy;

    using bicgstabl_t = numerical_algos::lin_solvers::bicgstabl<linear_op_t, precond_t, vec_ops_t, monitor_t, log_t>;
    using bicgstab_t = numerical_algos::lin_solvers::bicgstab<linear_op_t, precond_t, vec_ops_t, monitor_t, log_t>;
    
    using lin_solver_t = bicgstab_t;

    using time_step_const_t = time_steppers::time_step_adaptation_constant<vec_ops_t, log_t>;
    using time_step_err_ctrl_t = time_steppers::time_step_adaptation_error_control<vec_ops_t, log_t>;
    using time_step_adaptation_tolerance_t = time_steppers::time_step_adaptation_tolerance<vec_ops_t, log_t>;


    using single_step_const_t = time_steppers::explicit_time_step<vec_ops_t, nlin_op_t, log_t, time_step_const_t>;
    using single_step_err_ctrl_t = time_steppers::explicit_time_step<vec_ops_t, nlin_op_t, log_t, time_step_err_ctrl_t>;
    using single_step_implicit_const_t = time_steppers::implicit_time_step<vec_ops_t, nlin_op_t, linear_op_t, lin_solver_t, log_t, time_step_adaptation_tolerance_t>;

    using time_stepper_const_t = time_steppers::time_stepper<vec_ops_t, nlin_op_t, single_step_const_t,log_t>;
    using time_stepper_err_ctrl_t = time_steppers::time_stepper<vec_ops_t, nlin_op_t, single_step_err_ctrl_t,log_t>;
    using time_stepper_implicit_const_t = time_steppers::time_stepper<vec_ops_t, nlin_op_t, single_step_implicit_const_t, log_t>;

    if(argc != 4)
    {
        std::cout << argv[0] << " mu time name\n  mu - parameter, time - simulation time,\n";
        std::cout << "   name - name of the scheme:" << std::endl;
        time_steppers::detail::butcher_tables tables;
        auto names = tables.get_list_of_table_names();
        for(auto &n: names)
        {
            std::cout << n << " ";
        }
        std::cout << std::endl;        
        return(0);       
    }    
    real mu = std::stof(argv[1]);
    real simulation_time = std::stof(argv[2]);
    std::string scheme_name(argv[3]);

    log_t log, log3;
    log3.set_verbosity(0);
    vec_ops_t vec_ops(2);
    
    vec_t x0;

    vec_ops.init_vector(x0); vec_ops.start_use_vector(x0);


    nlin_op_t vdp;
    vdp.set_initial(x0);

    auto lin_op = &vdp.lin_op;
    auto prec = &vdp.prec;
    lin_solver_t lin_solver(&vec_ops, &log3);
    std::string implicit_scheme = "SDIRK3A3";


    time_step_const_t time_step_const(&vec_ops, &log, {0.0, simulation_time}, 1.0e-2);
    time_step_err_ctrl_t time_step_err_ctrl(&vec_ops, &log, {0.0, simulation_time});
    time_step_adaptation_tolerance_t time_step_adaptation_tolerance(&vec_ops, &log, {0.0, simulation_time} );
    time_step_adaptation_tolerance.set_adaptation_method("H211", 3);

    single_step_const_t explicit_step_const(&vec_ops, &time_step_const, &log, &vdp, mu, scheme_name);
    single_step_err_ctrl_t explicit_step_err_control(&vec_ops, &time_step_err_ctrl, &log, &vdp,  mu, scheme_name);
    single_step_implicit_const_t implicit_step_const(&vec_ops, &time_step_adaptation_tolerance, &log3, &vdp, lin_op, &lin_solver, mu, implicit_scheme);


    time_stepper_const_t time_stepper_const(&vec_ops, &vdp, &explicit_step_const, &log);
    time_stepper_err_ctrl_t time_stepper_err_ctrl(&vec_ops, &vdp, &explicit_step_err_control, &log);
    time_stepper_implicit_const_t time_stepper_implicit_const(&vec_ops, &vdp, &implicit_step_const, &log);


    log.info_f("executing explicit time stepper with time = %.2le", simulation_time);
    time_stepper_err_ctrl.set_parameter(mu);
    time_stepper_err_ctrl.set_initial_conditions(x0, 0.0);
    time_stepper_err_ctrl.execute();
    std::stringstream ss;
    ss << "vdp_result_" << scheme_name << ".dat";
    time_stepper_err_ctrl.save_norms( ss.str() );
    // std::stringstream ss;
    // ss << "x_" << simulation_time << "_sim.pos";
  
    log.info_f("executing implicit time stepper with time = %.2le", simulation_time);
    time_step_err_ctrl.reset_steps();
    //linsolver control
    unsigned int lin_solver_max_it = 10;
    real lin_solver_stiff_tol = 5.0e-10;
    unsigned int use_precond_resid = 1;
    unsigned int resid_recalc_freq = 1;
    unsigned int basis_sz = 1;

    lin_solver.set_preconditioner(prec);
    lin_solver.set_use_precond_resid(use_precond_resid);
    lin_solver.set_resid_recalc_freq(resid_recalc_freq);
    // lin_solver.set_basis_size(basis_sz);
    
    auto mon = &lin_solver.monitor();
    mon->init(lin_solver_stiff_tol, real(0.0), lin_solver_max_it);
    mon->set_save_convergence_history(true);
    mon->set_divide_out_norms_by_rel_base(true);    

    time_stepper_implicit_const.set_parameter(mu);
    time_stepper_implicit_const.set_initial_conditions(x0, 0.0);
    time_stepper_implicit_const.execute();
    std::stringstream ss1;
    ss1 << "vdp_result_" << implicit_scheme << ".dat";
    time_stepper_implicit_const.save_norms( ss1.str() );


    vec_ops.stop_use_vector(x0); vec_ops.free_vector(x0);
    return 0;
}

