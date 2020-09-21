#include <cmath>
#include <iostream>
#include <cstdio>
#include <string>

#include <utils/cuda_support.h>
#include <utils/log.h>
#include <external_libraries/cublas_wrap.h>
#include <external_libraries/cufft_wrap.h>


//problem dependant
#include <common/gpu_vector_operations.h>
#include <common/gpu_matrix_vector_operations.h>
#include <common/gpu_file_operations.h>
//problem dependant ends
//problem dependant
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/Kuramoto_Sivashinskiy_2D.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/linear_operator_KS_2D.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/preconditioner_KS_2D.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/convergence_strategy.h>
#include <nonlinear_operators/Kuramoto_Sivashinskiy_2D/system_operator.h>
//problem dependant ends

#include <numerical_algos/lin_solvers/default_monitor.h>
#include <numerical_algos/lin_solvers/bicgstabl.h>

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
        printf(" ==================================================================================================.\n");
        printf("Usage: %s path_to_config_file.json operaton, where:\n",argv[0]);
        printf("    path_to_config_file.json is the json file containing all project configuration;\n");  
        printf("    operaton stands for 'D', 'E', 'S', 'P' :\n");
        printf("    'D' - execute deflation-continuation.\n");             
        printf("    'E' - edit bifurcaiton curve.\n");             
        printf("    'S' - execute/edit stability curve.\n");             
        printf("    'P' - plot out the resutls.\n");             
        printf(" ==================================================================================================.\n");             
        return 0;
    }
    typedef SCALAR_TYPE real;
    std::string path_to_config_file_(argv[1]);
    char what_to_execute = argv[2][0];

    typedef main_classes::parameters<real> parameters_t;
    parameters_t parameters = main_classes::read_parameters_json<real>(path_to_config_file_);
    parameters.plot_all();


    unsigned int m_Krylov = parameters.stability_continuation.Krylov_subspace;
    real a_val = parameters.nonlinear_operator.problem_real_parameters_vector.at(0);
    real b_val = parameters.nonlinear_operator.problem_real_parameters_vector.at(1);
    int nvidia_pci_id = parameters.nvidia_pci_id;

    size_t Nx = parameters.nonlinear_operator.N_size.at(0);
    size_t Ny = parameters.nonlinear_operator.N_size.at(1);

//problem parameters ends

    typedef thrust::complex<real> complex;
    typedef gpu_vector_operations<real> vec_ops_real;

    typedef gpu_vector_operations<complex> vec_ops_complex;
    typedef gpu_vector_operations<real> vec_ops_real_im;
    typedef cufft_wrap_R2C<real> fft_t;
    typedef typename vec_ops_real::vector_type real_vec; 
    typedef typename vec_ops_complex::vector_type complex_vec;
    typedef typename vec_ops_real_im::vector_type real_im_vec;

    typedef gpu_matrix_vector_operations<real, real_im_vec> mat_vec_ops_t;
    typedef typename mat_vec_ops_t::matrix_type mat;


    typedef gpu_file_operations<vec_ops_real> files_real_t;
    typedef gpu_file_operations<vec_ops_real_im> files_real_im_t;

    typedef utils::log_std log_t;
    typedef numerical_algos::lin_solvers::default_monitor<
        vec_ops_real, log_t> monitor_t;
    typedef nonlinear_operators::Kuramoto_Sivashinskiy_2D<
        fft_t, 
        vec_ops_real, 
        vec_ops_complex, 
        vec_ops_real_im,
        Blocks_x_, 
        Blocks_y_> KS_2D_t;

    typedef nonlinear_operators::linear_operator_KS_2D<
        vec_ops_real_im, KS_2D_t> lin_op_t;
    
    typedef nonlinear_operators::preconditioner_KS_2D<
        vec_ops_real_im, KS_2D_t, lin_op_t> prec_t;

    typedef container::knots<real> knots_t;


    init_cuda(nvidia_pci_id); // )(PCI) where PCI is the GPU PCI ID


    fft_t *CUFFT_C2R = new fft_t(Nx, Ny);
    size_t My=CUFFT_C2R->get_reduced_size();
    cublas_wrap *CUBLAS = new cublas_wrap();
    CUBLAS->set_pointer_location_device(false);
    vec_ops_real vec_ops_R(Nx*Ny, CUBLAS);
    vec_ops_complex vec_ops_C(Nx*My, CUBLAS);
    vec_ops_real_im vec_ops_R_im(Nx*My-1, CUBLAS);

    mat_vec_ops_t mat_vec_ops(Nx*My-1, m_Krylov, CUBLAS);


    files_real_t file_ops( (vec_ops_real*) &vec_ops_R);
    files_real_im_t file_ops_im( (vec_ops_real_im*) &vec_ops_R_im);    
    //CUDA GRIDS
    dim3 Blocks; dim3 Grids; dim3 Grids_F;
    KS_2D_t KS2D(a_val, b_val, Nx, Ny, (vec_ops_real*) &vec_ops_R, (vec_ops_complex*) &vec_ops_C, (vec_ops_real_im*) &vec_ops_R_im, CUFFT_C2R);
    KS2D.get_cuda_grid(Grids, Grids_F, Blocks);
    printf("Blocks = (%i,%i,%i)\n", Blocks.x, Blocks.y, Blocks.z);
    printf("Grids = (%i,%i,%i)\n", Grids.x, Grids.y, Grids.z);
    printf("GridsFourier = (%i,%i,%i)\n", Grids_F.x, Grids_F.y, Grids_F.z);


    log_t log;
    log_t log_linsolver;
    log_linsolver.set_verbosity(1);  


    if( (what_to_execute=='D')||(what_to_execute == 'E') )
    {

        typedef main_classes::deflation_continuation<vec_ops_real_im, files_real_im_t, log_t, monitor_t, KS_2D_t, lin_op_t, prec_t, numerical_algos::lin_solvers::bicgstabl, nonlinear_operators::system_operator, parameters_t> deflation_continuation_t;

        deflation_continuation_t DC( (vec_ops_real_im*) &vec_ops_R_im, (files_real_im_t*) &file_ops_im, (log_t*) &log, (log_t*) &log_linsolver, (KS_2D_t*) &KS2D, (parameters_t*) &parameters);

        //set all parameters from the structure
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
    else if(what_to_execute == 'S')
    {
        typedef main_classes::stability_continuation<vec_ops_real_im, mat_vec_ops_t, files_real_im_t, log_t, monitor_t, KS_2D_t, lin_op_t, prec_t, numerical_algos::lin_solvers::bicgstabl, nonlinear_operators::system_operator, parameters_t> stability_t;

        vec_ops_real_im vec_ops_small(m_Krylov, CUBLAS);
        mat_vec_ops_t mat_ops_small(m_Krylov, m_Krylov, CUBLAS);

        stability_t ST( (vec_ops_real_im*) &vec_ops_R_im, (mat_vec_ops_t*) &mat_vec_ops, (vec_ops_real_im*) &vec_ops_small, (mat_vec_ops_t*) &mat_ops_small, (files_real_im_t*) &file_ops_im, (log_t*) &log, (log_t*) &log_linsolver, (KS_2D_t*) &KS2D, (parameters_t*) &parameters);

        ST.set_parameters();
        ST.edit();
        ST.execute();
    }
    else if(what_to_execute == 'P')
    {
        typedef main_classes::plot_diagram_to_pos<vec_ops_real_im, mat_vec_ops_t, files_real_im_t, log_t, monitor_t, KS_2D_t, lin_op_t, prec_t, numerical_algos::lin_solvers::bicgstabl, nonlinear_operators::system_operator, parameters_t> plot_diagram_t;       

        plot_diagram_t PD( (vec_ops_real_im*) &vec_ops_R_im, (files_real_im_t*) &file_ops_im, (log_t*) &log, (log_t*) &log_linsolver, (KS_2D_t*) &KS2D, (parameters_t*) &parameters);

        PD.set_parameters();
        PD.execute();
    }
    else
    {
        std::cout << "No correct usage scheme was selected." << std::endl;
    }


    return 0;
}
