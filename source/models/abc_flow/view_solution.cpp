#include <iostream>
#include <cstdio>
#include <string>

#include <common/macros.h>
#include <external_libraries/cufft_wrap.h>
#include <external_libraries/cublas_wrap.h>
#include <utils/cuda_support.h>
#include <nonlinear_operators/abc_flow/abc_flow.h>
#include <common/gpu_vector_operations.h>
#include <common/gpu_file_operations.h>

#define Blocks_x_ 32
#define Blocks_y_ 16


int main(int argc, char const *argv[])
{
    if(argc!=4)
    {
        printf("=======================================================\n");
        printf("Usage: %s N input_file output_file_names, where:\n", argv[0]);
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

    typedef nonlinear_operators::abc_flow<cufft_type, 
            gpu_vector_operations_real_t, 
            gpu_vector_operations_complex_t, 
            gpu_vector_operations_t,
            Blocks_x_, Blocks_y_> abc_flow_t;
    typedef gpu_file_operations<gpu_vector_operations_t> gpu_file_operations_t;


    size_t N = std::atoi(argv[1]);
    std::string input_file_name(argv[2]);
    std::string output_file_name_prefix(argv[3]);

    size_t Nx = N;
    size_t Ny = N;
    size_t Nz = N;
    std::cout << "Plotting solution file " << input_file_name << " with prefix " <<  output_file_name_prefix << " ." << std::endl;
    std::cout << "Using discretization: " << Nx << "X" << Ny << "X" << Nz << std::endl;

    cufft_type *CUFFT_C2R = new cufft_type(Nx, Ny, Nz);
    size_t Mz = CUFFT_C2R->get_reduced_size();
    size_t Nv = real(6*(Nx*Ny*Mz-1));

    size_t scale = 2;
    size_t Nx_vis = Nx*scale;
    size_t Ny_vis = Ny*scale;
    size_t Nz_vis = Nz*scale;
    cufft_type *CUFFT_C2R_vis = new cufft_type(Nx_vis, Ny_vis, Nz_vis);
    size_t Mz_vis = CUFFT_C2R_vis->get_reduced_size();

    cublas_wrap *CUBLAS = new cublas_wrap(true);
    CUBLAS->set_pointer_location_device(false);
    gpu_vector_operations_real_t *vec_ops_R = new gpu_vector_operations_real_t(Nx*Ny*Nz, CUBLAS);
    gpu_vector_operations_complex_t *vec_ops_C = new gpu_vector_operations_complex_t(Nx*Ny*Mz, CUBLAS);
    gpu_vector_operations_t *vec_ops = new gpu_vector_operations_t(Nv, CUBLAS);
    gpu_file_operations_t *file_ops = new gpu_file_operations_t(vec_ops);

    abc_flow_t *abc = new abc_flow_t(Nx, Ny, Nz, vec_ops_R, vec_ops_C, vec_ops, CUFFT_C2R);

    vec b;



    vec_ops->init_vector(b); vec_ops->start_use_vector(b);
    try
    {
        file_ops->read_vector(input_file_name, b);
        std::string f_name_vec(output_file_name_prefix +"_vec.pos");
        std::string f_name_abs(output_file_name_prefix +"_abs.pos");      
        abc->write_solution_vec(f_name_vec, (vec&)b);
        std::cout << "wrote " << f_name_vec << std::endl;
        abc->write_solution_abs(f_name_abs, (vec&)b);
        std::cout << "wrote " << f_name_abs << std::endl;
        
        abc->write_solution_scaled(CUFFT_C2R_vis, Nx_vis, Ny_vis, Nz_vis, output_file_name_prefix+"_s_vec.pos", output_file_name_prefix+"_s_abs.pos", (vec&)b);

        std::cout << "wrote scaled files: " << output_file_name_prefix+"_s_vec.pos" << " " << output_file_name_prefix+"_s_abs.pos" <<  std::endl;   
    }
    catch(...)
    {
        std::cout << "File " << input_file_name << " not found!" << std::endl;
        vec_ops->stop_use_vector(b); vec_ops->free_vector(b);
        return 0;
    }

    vec_ops->stop_use_vector(b); vec_ops->free_vector(b);


    delete abc;
    delete file_ops;
    delete vec_ops_R;
    delete vec_ops_C;
    delete vec_ops;    
    delete CUBLAS;
    delete CUFFT_C2R_vis;
    delete CUFFT_C2R;



    return 0;
}