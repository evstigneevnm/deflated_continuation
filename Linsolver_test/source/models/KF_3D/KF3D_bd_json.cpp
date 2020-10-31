#include <cmath>
#include <iostream>
#include <cstdio>
#include <string>

#include <utils/cuda_support.h>
#include <utils/log.h>
#include <external_libraries/cublas_wrap.h>
#include <external_libraries/cufft_wrap.h>


//vector dependant
#include <common/gpu_vector_operations.h>
#include <common/gpu_matrix_vector_operations.h>
#include <common/gpu_file_operations.h>
//vector dependant ends
//problem dependant
#include <nonlinear_operators/Kolmogorov_flow_3D/Kolmogorov_3D.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/linear_operator_K_3D.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/preconditioner_K_3D.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/convergence_strategy.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/system_operator.h>
//problem dependant ends

#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/bicgstabl.h>
// #include <numerical_algos/lin_solvers/gmres.h>

#include <main/deflation_continuation.hpp>
#include <main/stability_continuation.hpp>
#include <main/plot_diagram_to_pos.hpp>
#include <main/parameters.hpp>


#ifndef Blocks_x_
    #define Blocks_x_ 32
#endif
#ifndef Blocks_y_
    #define Blocks_y_ 16
#endif

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
        printf("    'S' - execute/edit stability curve.\n");             
        printf("    'P' - plot out the resutls.\n");             
        printf(".==================================================================================================.\n");             
        return 0;
    }
    typedef SCALAR_TYPE real;
    std::string path_to_config_file_(argv[1]);
    char what_to_execute = argv[2][0];

    typedef main_classes::parameters<real> parameters_t;
    parameters_t parameters = main_classes::read_parameters_json<real>(path_to_config_file_);
    parameters.plot_all();


    unsigned int m_Krylov = parameters.stability_continuation.Krylov_subspace;
    real alpha = parameters.nonlinear_operator.problem_real_parameters_vector.at(0);
    int one_over_alpha = parameters.nonlinear_operator.problem_int_parameters_vector.at(0);
    int nvidia_pci_id = parameters.nvidia_pci_id;

    size_t Nx = parameters.nonlinear_operator.N_size.at(0);
    size_t Ny = parameters.nonlinear_operator.N_size.at(1);
    size_t Nz = parameters.nonlinear_operator.N_size.at(2);

    
    typedef utils::log_std log_t;
    typedef thrust::complex<real> complex;
    typedef gpu_vector_operations<real> vec_ops_real_t;
    typedef gpu_vector_operations<complex> vec_ops_complex_t;
    typedef gpu_vector_operations<real> vec_ops_t;
    typedef cufft_wrap_R2C<real> cufft_type;
    typedef typename vec_ops_t::vector_type vec_t;

    typedef gpu_matrix_vector_operations<real, vec_t> mat_vec_ops_t;
    
    typedef gpu_file_operations<vec_ops_t> files_t;

    typedef numerical_algos::lin_solvers::default_monitor<
        vec_ops_t,log_t> monitor_t;
    
    typedef nonlinear_operators::Kolmogorov_3D<cufft_type, 
            vec_ops_real_t, 
            vec_ops_complex_t, 
            vec_ops_t,
            Blocks_x_, Blocks_y_> KF_3D_t;

    typedef nonlinear_operators::linear_operator_K_3D<
        vec_ops_t, KF_3D_t> lin_op_t;
    
    typedef nonlinear_operators::preconditioner_K_3D<
        vec_ops_t, KF_3D_t, lin_op_t> prec_t;

    typedef container::knots<real> knots_t;


    init_cuda(nvidia_pci_id); 


    cufft_type *CUFFT_C2R = new cufft_type(Nx, Ny, Nz);
    size_t Mz=CUFFT_C2R->get_reduced_size();
    size_t Nv = 3*(Nx*Ny*Mz-1);

    cublas_wrap *CUBLAS = new cublas_wrap();
    CUBLAS->set_pointer_location_device(false);
    vec_ops_real_t vec_ops_R(Nx*Ny*Nz, CUBLAS);
    vec_ops_complex_t vec_ops_C(Nx*Ny*Mz, CUBLAS);
    vec_ops_t vec_ops(Nv, CUBLAS);
    mat_vec_ops_t mat_vec_ops(Nv, m_Krylov, CUBLAS);

    files_t file_ops_im( (vec_ops_t*) &vec_ops);    

    KF_3D_t KF3D(alpha, Nx, Ny, Nz, (vec_ops_real_t*) &vec_ops_R, (vec_ops_complex_t*) &vec_ops_C, (vec_ops_t*) &vec_ops, CUFFT_C2R);

    log_t log;
    log_t log_linsolver;
    log_linsolver.set_verbosity(1);
       

    if( (what_to_execute=='D')||(what_to_execute == 'E') )
    {
        typedef main_classes::deflation_continuation<
            vec_ops_t, files_t, log_t, monitor_t, KF_3D_t, 
            lin_op_t, prec_t, numerical_algos::lin_solvers::bicgstabl, 
            nonlinear_operators::system_operator, parameters_t> deflation_continuation_t;

        deflation_continuation_t DC( (vec_ops_t*) &vec_ops, (files_t*) &file_ops_im, (log_t*) &log,  (log_t*) &log_linsolver, (KF_3D_t*) &KF3D, (parameters_t*) &parameters);
        DC.set_parameters();
        if(what_to_execute == 'D')
        {
            DC.use_analytical_solution(true);
            DC.execute();
        }
        else if(what_to_execute == 'E')
        {
            DC.edit();
        }

    }
    else if(what_to_execute == 'S')
    {
        typedef main_classes::stability_continuation<vec_ops_t, mat_vec_ops_t, files_t, log_t, monitor_t, KF_3D_t, lin_op_t, prec_t, numerical_algos::lin_solvers::bicgstabl, nonlinear_operators::system_operator, parameters_t> stability_t;

        vec_ops_t vec_ops_small(m_Krylov, CUBLAS);
        mat_vec_ops_t mat_ops_small(m_Krylov, m_Krylov, CUBLAS);

        stability_t ST( (vec_ops_t*) &vec_ops, (mat_vec_ops_t*) &mat_vec_ops, (vec_ops_t*) &vec_ops_small, (mat_vec_ops_t*) &mat_ops_small, (files_t*) &file_ops_im, (log_t*) &log, (log_t*) &log_linsolver, (KF_3D_t*) &KF3D, (parameters_t*) &parameters);

        ST.set_parameters();
        ST.edit();
        ST.execute();        
    }
    else if(what_to_execute == 'P')
    {
        typedef main_classes::plot_diagram_to_pos<vec_ops_t, mat_vec_ops_t, files_t, log_t, monitor_t, KF_3D_t, lin_op_t, prec_t, numerical_algos::lin_solvers::bicgstabl, nonlinear_operators::system_operator, parameters_t> plot_diagram_t;       

        plot_diagram_t PD( (vec_ops_t*) &vec_ops, (files_t*) &file_ops_im, (log_t*) &log, (log_t*) &log_linsolver, (KF_3D_t*) &KF3D, (parameters_t*) &parameters);

        PD.set_parameters();
        PD.execute();
    }
    else
    {
        std::cout << "No correct usage scheme was selected." << std::endl;
    }    
    
    return 0;
}
