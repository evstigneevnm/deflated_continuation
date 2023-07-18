#include <utils/log.h>
#include <common/cpu_vector_operations.h>
#include "../glued_poincare_map_linear_operator.h"
#include "rossler_operator.h"
#include <time_stepper/explicit_time_step.h>
#include <time_stepper/time_step_adaptation_error_control.h>
#include <time_stepper/time_step_adaptation_constant.h>
#include <periodic_orbit/hyperplane.h>
#include <periodic_orbit/time_stepper_to_section.h>
#include <periodic_orbit/poincare_map_operator.h>


int main(int argc, char const *argv[])
{
    using log_t = utils::log_std;
    using real = SCALAR_TYPE;
    using T = real;
    using vec_ops_t = cpu_vector_operations<real>;
    using vec_t = typename vec_ops_t::vector_type;
    using nlin_op_t = nonlinear_operators::rossler<vec_ops_t>;
    using hyperplane_t = periodic_orbit::hyperplane<vec_ops_t, nlin_op_t>;
    // template<class VectorOperations, class NonlinearOperator, class TimeStepAdaptation, template<class, class, class, class> class SingleStepper, class Hyperplane, class Log >
    using glued_poincare_map_linear_op_t = periodic_orbit::glued_poincare_map_linear_operator<vec_ops_t, nlin_op_t,  time_steppers::time_step_adaptation_error_control, time_steppers::explicit_time_step,  hyperplane_t, log_t>;
    
    using poincare_map_operator_t = periodic_orbit::poincare_map_operator<vec_ops_t, nlin_op_t, time_steppers::time_step_adaptation_error_control, time_steppers::explicit_time_step, hyperplane_t, log_t>;

    log_t log;


    std::string scheme_name("RKDP45");
    if(argc == 2)
    {
        scheme_name = std::string(argv[1]);
        if((scheme_name == "-h")||(scheme_name == "--help"))
        {
            std::cout << "Usage: " << argv[0] << " scheme_name" << std::endl;
            std::cout << "scheme_name: EE, HE, RK33SSP, RK43SSP, RKDP45, RK64SSP" << std::endl;
            return 1;
        }
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

    T ref_error = std::numeric_limits<T>::epsilon();


    log.info("test glued poincate linear operator.");
    vec_ops_t vec_ops(3);
    nlin_op_t rossler(&vec_ops, 2, 0.2, 0.2, 5.7);
    poincare_map_operator_t poincare_map(&vec_ops, &rossler, &log, 1000.0, 5.7, method);
    glued_poincare_map_linear_op_t poincare_map_x(&vec_ops, &rossler, &log, 1000.0, 5.7, method);
    
    vec_t xb, x0, v0, v1;
    vec_ops.init_vectors(xb,x0,v0,v1); vec_ops.start_use_vectors(xb,x0,v0,v1);
    rossler.set_period_point(xb);
    rossler.set_period_point(x0);
    v0[0] = 1.0;
    v0[1] = 2.0;
    v0[2] = 3.0;
    
    
    hyperplane_t hyperplane(&vec_ops, &rossler, x0, 5.7);
    
    log.info("running poincare map");
    poincare_map.set_parameter(5.7);
    poincare_map.set_hyperplanes({&hyperplane,&hyperplane});
    poincare_map.F(xb, 5.7, xb);

    log.info("running poincare map_x");
    poincare_map_x.set_parameter(5.7);
    poincare_map_x.set_hyperplanes({&hyperplane,&hyperplane});
    poincare_map_x.apply(v0, v1);

    std::cout << "T_period map = " << poincare_map.get_period_estmate_time() << " T_period map_x = " << poincare_map_x.get_period_estmate_time() << std::endl;

    poincare_map.save_period_estmate_norms("rossler_poincare_map.dat");
    poincare_map_x.save_period_estmate_norms("rossler_poincare_map_x.dat");

    for(int j=0;j<3;j++)
    {
        log.info_f("%le -> %le", x0[j], xb[j]);
    }


    real err = 0.0;
    err += (v1[0]-(-17.1938638231880*0))*(v1[0]-(-17.1938638231880*0));
    err += (v1[1]-(35.3045017642549))*(v1[1]-(35.3045017642549));
    err += (v1[2]-(-0.199172394970001))*(v1[2]-(-0.199172394970001));
    
    err = std::sqrt(err);


    for(int j=0;j<3;j++)
    {
        log.info_f("%le -> %le", v0[j], v1[j]);
    }
    

    vec_ops.stop_use_vectors(xb,x0,v0,v1); vec_ops.free_vectors(xb,x0,v0,v1);

    int N_tests = 1;
    const uint8_t N = 3;
    auto ref_error_val = ref_error*N_tests*std::sqrt(N);
    if(err > ref_error*N_tests*std::sqrt(N) )
    {
        log.error_f("Got error = %e with reference = %e", err,  ref_error_val);
        return 1;
    }
    else
    {
        log.info_f("No errors with reference = %e", ref_error_val);
        return 0;    
    }


	return 0;
}