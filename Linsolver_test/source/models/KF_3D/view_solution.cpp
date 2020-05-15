#include <iostream>
#include <cstdio>
#include <string>

#include <common/macros.h>
#include <external_libraries/cufft_wrap.h>
#include <external_libraries/cublas_wrap.h>
#include <utils/cuda_support.h>
#include <nonlinear_operators/Kolmogorov_flow_3D/Kolmogorov_3D.h>
#include <common/gpu_vector_operations.h>
#include <common/gpu_file_operations.h>

#define Blocks_x_ 32
#define Blocks_y_ 16


int main(int argc, char const *argv[])
{
    if(argc!=5)
    {
        printf("=======================================================\n");
        printf("Usage: %s alpha N input_file output_file_names, where:\n", argv[0]);
        printf("    0<alpha<=1 ;\n");
        printf("    N is the discretization in one 2\\pi direction;\n");
        printf("    input_file is the file name of the column vector of the probelm;\n");
        printf("    output_file_names is the name of files withough extension to be placed, i.e:\n"); 
        printf("     if output_file_names = 'fff', then the program will generate files fff_vec.pos and fff_abs.pos.\n");
        printf("=======================================================\n");
        return 0;
    }

    typedef SCALAR_TYPE real;

    typedef thrust::complex<real> complex;
    typedef gpu_vector_operations<real> gpu_vector_operations_real_t;
    typedef gpu_vector_operations<complex> gpu_vector_operations_complex_t;
    typedef gpu_vector_operations<real> gpu_vector_operations_t;
    typedef cufft_wrap_R2C<real> cufft_type;
    
    typedef typename gpu_vector_operations_real_t::vector_type real_vec; 
    typedef typename gpu_vector_operations_complex_t::vector_type complex_vec;
    typedef typename gpu_vector_operations_t::vector_type vec;

    typedef nonlinear_operators::Kolmogorov_3D<cufft_type, 
            gpu_vector_operations_real_t, 
            gpu_vector_operations_complex_t, 
            gpu_vector_operations_t,
            Blocks_x_, Blocks_y_> KF_3D_t;
    typedef gpu_file_operations<gpu_vector_operations_t> gpu_file_operations_t;


    real alpha = std::atof(argv[1]);
    size_t N = std::atoi(argv[2]);
    int one_over_alpha = int(1/alpha);
    std::string input_file_name(argv[3]);
    std::string output_file_name_prefix(argv[4]);

    size_t Nx = N*one_over_alpha;
    size_t Ny = N;
    size_t Nz = N;
    std::cout << "Plotting solution file " << input_file_name << " with prefix " <<  output_file_name_prefix << " ." << std::endl;
    std::cout << "Using alpha = " << alpha << ", with discretization: " << Nx << "X" << Ny << "X" << Nz << std::endl;

    cufft_type *CUFFT_C2R = new cufft_type(Nx, Ny, Nz);
    size_t Mz = CUFFT_C2R->get_reduced_size();
    size_t Nv = real(3*(Nx*Ny*Mz-1));


    cublas_wrap *CUBLAS = new cublas_wrap(true);
    CUBLAS->set_pointer_location_device(false);
    gpu_vector_operations_real_t *vec_ops_R = new gpu_vector_operations_real_t(Nx*Ny*Nz, CUBLAS);
    gpu_vector_operations_complex_t *vec_ops_C = new gpu_vector_operations_complex_t(Nx*Ny*Mz, CUBLAS);
    gpu_vector_operations_t *vec_ops = new gpu_vector_operations_t(Nv, CUBLAS);
    gpu_file_operations_t *file_ops = new gpu_file_operations_t(vec_ops);

    KF_3D_t *KF_3D = new KF_3D_t(alpha, Nx, Ny, Nz, vec_ops_R, vec_ops_C, vec_ops, CUFFT_C2R);

    vec b;

    vec_ops->init_vector(b); vec_ops->start_use_vector(b);
    try
    {
        file_ops->read_vector(input_file_name, b);
        std::string f_name_vec(output_file_name_prefix +"_vec.pos");
        std::string f_name_abs(output_file_name_prefix +"_abs.pos");      
        KF_3D->write_solution_vec(f_name_vec, (vec&)b);
        std::cout << "wrote " << f_name_vec << std::endl;
        KF_3D->write_solution_abs(f_name_abs, (vec&)b);
        std::cout << "wrote " << f_name_abs << std::endl;
    }
    catch(...)
    {
        std::cout << "File " << input_file_name << " not found!" << std::endl;
        return 0;
    }

    vec_ops->stop_use_vector(b); vec_ops->free_vector(b);


    delete KF_3D;
    delete file_ops;
    delete vec_ops_R;
    delete vec_ops_C;
    delete vec_ops;    
    delete CUBLAS;
    delete CUFFT_C2R;


    return 0;
}