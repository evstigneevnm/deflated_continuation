#include <cmath>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sstream>

#include <utils/init_cuda.h>
#include <external_libraries/cublas_wrap.h>
#include <external_libraries/lapack_wrap.h>
#include <utils/log.h>


//vector dependant
#include <common/gpu_vector_operations.h>
#include <common/gpu_matrix_vector_operations.h>
#include <common/gpu_file_operations.h>
//vector dependant ends
//problem dependant
#include <nonlinear_operators/overscreening_breakdown/overscreening_breakdown.h>
#include <nonlinear_operators/overscreening_breakdown/linear_operator_overscreening_breakdown.h>
#include <nonlinear_operators/overscreening_breakdown/preconditioner_overscreening_breakdown.h>
#include <nonlinear_operators/overscreening_breakdown/linear_operator_overscreening_breakdown_shifted.h>
#include <nonlinear_operators/overscreening_breakdown/preconditioner_overscreening_breakdown_shifted.h>
#include <nonlinear_operators/overscreening_breakdown/system_operator.h>
#include <nonlinear_operators/overscreening_breakdown/convergence_strategy.h>
//problem dependant ends

#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/exact_wrapper.h>

#include <main/deflation_continuation.hpp>
#include <main/plot_diagram_to_pos.hpp>
#include <main/parameters.hpp>

template<class T>
struct params_s
{
    size_t N = 10;
    T sigma = 1.0;
    T L = 1.0;
    T gamma = 1.0;
    T delta = 1.0;    
    T mu = 1.0;
    T u0 = 1.0;

    void print_data() const
    {
        std::cout << "=== params_s: " << std::endl;
        std::cout << "=   N = " << N << std::endl;
        std::cout << "=   sigma = " << sigma << std::endl;
        std::cout << "=   L = " << L << std::endl;
        std::cout << "=   gamma = " << gamma << std::endl;
        std::cout << "=   delta = " << delta << std::endl;
        std::cout << "=   mu = " << mu << std::endl;
        std::cout << "=   u0 = " << u0 << std::endl;
        std::cout << "=   .........." << std::endl;
    }
    params_s(size_t N_p, T sigma_p, const std::vector<T>& other_params_p):
    N(N_p),
    sigma(sigma_p)
    {
        L = other_params_p.at(0);
        gamma = other_params_p.at(1);
        delta = other_params_p.at(2);
        mu = other_params_p.at(3);
        u0 = other_params_p.at(4);

        print_data();
    }
};

int main(int argc, char const *argv[])
{
    
    if(argc!=3)
    {
        printf(".==================================================================================================.\n");
        printf("Usage: %s path_to_config_file.json operaton, where:\n",argv[0]);
        printf("    path_to_config_file.json is the json file containing all project configuration;\n");  
        printf("    operaton stands for 'D', 'E', 'S', 'P' :\n");
        printf("    'D' - execute deflation-continuation.\n");             
        printf("    'E' - edit bifurcaiton curve.\n");                       
        printf("    'P' - plot out the resutls.\n");             
        printf(".==================================================================================================.\n");             
        return 0;
    }
    typedef SCALAR_TYPE real;
    using params_st = params_s<real>;

    std::string path_to_config_file_(argv[1]);
    char what_to_execute = argv[2][0];

    typedef main_classes::parameters<real> parameters_t;
    parameters_t parameters = main_classes::read_parameters_json<real>(path_to_config_file_);
    parameters.plot_all();


    unsigned int m_Krylov = parameters.stability_continuation.Krylov_subspace;
    
    int nvidia_pci_id = parameters.nvidia_pci_id;
    bool use_high_precision_reduction = parameters.use_high_precision_reduction;

    size_t N = parameters.nonlinear_operator.N_size.at(0);
    params_st problem_params(N, 0.05, parameters.nonlinear_operator.problem_real_parameters_vector);
    
    using log_t = utils::log_std ;

    using vec_ops_t = gpu_vector_operations<real>;

    using vec_t = typename vec_ops_t::vector_type;

    using mat_ops_t = gpu_matrix_vector_operations<real, vec_t>;
    
    using files_t = gpu_file_operations<vec_ops_t>;

    using monitor_t = numerical_algos::lin_solvers::default_monitor<vec_ops_t,log_t>;
    
    using ob_prob_t = nonlinear_operators::overscreening_breakdown<vec_ops_t, mat_ops_t>;

    //standard linear operators and preconditioners
    //linear operators and preconditioners with shifts
    using lin_op_t = nonlinear_operators::linear_operator_overscreening_breakdown<vec_ops_t, mat_ops_t, ob_prob_t>;
    using lin_op_shifted_t = nonlinear_operators::linear_operator_overscreening_breakdown_shifted<vec_ops_t, mat_ops_t, ob_prob_t>;
    using prec_t = nonlinear_operators::preconditioner_overscreening_breakdown<vec_ops_t, mat_ops_t, ob_prob_t, lin_op_t>;
    using prec_shifted_t = nonlinear_operators::preconditioner_overscreening_breakdown_shifted<vec_ops_t, mat_ops_t, ob_prob_t, lin_op_shifted_t>;


    using knots_t = container::knots<real>;


    utils::init_cuda(nvidia_pci_id);

    cublas_wrap cublas(true);
    vec_ops_t vec_ops(N, &cublas);
    mat_ops_t mat_ops(vec_ops.get_vector_size(), vec_ops.get_vector_size(), vec_ops.get_cublas_ref() );
    files_t vec_file_ops(&vec_ops);
    ob_prob_t ob_prob(&vec_ops, &mat_ops, problem_params );
    if(use_high_precision_reduction)
    {
        vec_ops.use_high_precision();
    }

    log_t log;
    log_t log_linsolver;
    log_linsolver.set_verbosity(1);
       

    if( (what_to_execute=='D')||(what_to_execute == 'E') )
    {
        // typedef main_classes::deflation_continuation<
        //     vec_ops_t, files_t, log_t, monitor_t, KF_3D_t, 
        //     lin_op_t, prec_t, numerical_algos::lin_solvers::bicgstabl, 
        //     nonlinear_operators::system_operator, parameters_t> deflation_continuation_t;

        using deflation_continuation_t = main_classes::deflation_continuation<
            vec_ops_t, files_t, log_t, monitor_t, ob_prob_t, 
            lin_op_t, prec_t, numerical_algos::lin_solvers::exact_wrapper, 
            nonlinear_operators::system_operator, parameters_t>;

        deflation_continuation_t DC( (vec_ops_t*) &vec_ops, (files_t*) &vec_file_ops, (log_t*) &log,  (log_t*) &log_linsolver, (ob_prob_t*) &ob_prob, (parameters_t*) &parameters);
        DC.set_parameters();
        if(what_to_execute == 'D')
        {
            DC.use_analytical_solution(false);
            DC.execute();
        }
        else if(what_to_execute == 'E')
        {
            DC.edit();
        }

    }
    else if(what_to_execute == 'P')
    {
        // typedef main_classes::plot_diagram_to_pos<vec_ops_t, mat_vec_ops_t, files_t, log_t, monitor_t, KF_3D_t, lin_op_t, prec_t, numerical_algos::lin_solvers::bicgstabl, nonlinear_operators::system_operator, parameters_t> plot_diagram_t;       
        
        // plot_diagram_t PD( (vec_ops_t*) &vec_ops, (files_t*) &file_ops_im, (log_t*) &log, (log_t*) &log_linsolver, (KF_3D_t*) &KF3D, (parameters_t*) &parameters);
        
        // PD.set_parameters();
        // PD.execute();
    }
    else
    {
        std::cout << "No correct usage scheme was selected." << std::endl;
    }    
    return 0;
}
