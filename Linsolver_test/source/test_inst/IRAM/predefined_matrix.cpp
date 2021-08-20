#include <common/gpu_vector_operations.h>
#include <common/gpu_matrix_vector_operations.h>
#include <common/gpu_matrix_file_operations.h>
#include <utils/cuda_support.h>
#include <utils/log.h>
#include <external_libraries/cublas_wrap.h>
#include <external_libraries/lapack_wrap.h>
#include <stability/IRAM/iram_container.h>
#include <stability/IRAM/shift_bulge_chase.h>
#include <common/gpu_file_operations_functions.h>

template<class T>
void print_matrix(int Nrows, int Ncols, T* A)
{
    T* A_h = new T[Nrows*Nrows];
    device_2_host_cpy(A_h, A, Nrows*Nrows);

    for(int j =0;j<Nrows;j++)
    {
        for(int k=0;k<Ncols;k++)
        {
            std::cout << A_h[j+Nrows*k] << " ";
        }
        std::cout << std::endl;
    }

    delete [] A_h;
}
template<class T>
void print_matrix(int Nl, T* A)
{
    T* A_h = new T[Nl*Nl];
    device_2_host_cpy(A_h, A, Nl*Nl);    
    for(int j =0;j<Nl;j++)
    {
        for(int k=0;k<Nl;k++)
        {
            std::cout << A_h[j+Nl*k] << " ";
        }
        std::cout << std::endl;
    }
    delete [] A_h;
}   

template<class T>
void print_vector(int Ncols, T* v)
{
    T* v_h = new T[Ncols];
    device_2_host_cpy(v_h, v, Ncols);
    for(int k=0;k<Ncols;k++)
    {
        std::cout << v[k] << std::endl;
    }
    delete [] v_h;

}



int main(int argc, char const *argv[])
{
    using real = SCALAR_TYPE;
    using vec_ops_t = gpu_vector_operations<real>;
    using T_vec = typename vec_ops_t::vector_type;
    using mat_ops_t = gpu_matrix_vector_operations<real, T_vec>;
    using T_mat = typename mat_ops_t::matrix_type;
    using mat_files_t = gpu_matrix_file_operations<mat_ops_t>;
    using lapack_wrap_t = lapack_wrap<real>;
    using log_t = utils::log_std;
    using container_t = stability::IRAM::iram_container<vec_ops_t,mat_ops_t,log_t>;
    using bulge_t = stability::IRAM::shift_bulge_chase<vec_ops_t, mat_ops_t, lapack_wrap_t, log_t>;

    if(argc != 3)
    {
        std::cout << "Usage: " << argv[0] << " V_matrix H_matrix" << std::endl;  
        return 0; 
    }
    std::string file_nameV(argv[1]);
    std::string file_nameH(argv[2]);

    if(init_cuda(4) == 0)
    {
        return 0;
    }

    unsigned int m = 40;
    unsigned int k = 10;
    size_t N = 563;


    cublas_wrap CUBLAS(true);
    lapack_wrap_t blas(m);
    log_t log;
    vec_ops_t vec_ops_N(N, &CUBLAS);
    vec_ops_t vec_ops_m(m, &CUBLAS);
    mat_ops_t mat_ops_N(N, m, &CUBLAS);
    mat_ops_t mat_ops_m(m, m, &CUBLAS);
    mat_files_t mat_f_m(&mat_ops_m);
    mat_files_t mat_f_N(&mat_ops_N);

    T_mat Q_deb;
    T_mat H_deb;
    T_vec f_deb;
    mat_ops_m.init_matrix(Q_deb); mat_ops_m.start_use_matrix(Q_deb);
    mat_ops_m.init_matrix(H_deb); mat_ops_m.start_use_matrix(H_deb);
    vec_ops_N.init_vector(f_deb); vec_ops_N.start_use_vector(f_deb);
    
    container_t container(&vec_ops_N, &mat_ops_N, &vec_ops_m, &mat_ops_m, &log);
    bulge_t bulge(&vec_ops_N, &mat_ops_N, &vec_ops_m, &mat_ops_m, &log, &blas);
    bulge.set_number_of_desired_eigenvalues(k);

    container.force_gpu();
    mat_f_m.read_matrix(file_nameH, container.ref_H() );
    mat_f_N.read_matrix(file_nameV, container.ref_V() );
    container.set_f();

    container.to_cpu();
    container.to_gpu();

    bulge.set_target("LM");
    // bulge.select_shifts(container.ref_H() );
    // bulge.form_polynomial(container.ref_H() );


    // container.to_gpu();
    // mat_f_m.read_matrix("dat_files/iram/Q_test.dat", Q_deb );
    // mat_f_m.read_matrix("dat_files/iram/H1_test.dat", H_deb );
    // gpu_file_operations_functions::read_vector<real>("dat_files/iram/f_test.dat", N, container.ref_f() );
    // bulge._debug_set_Q(Q_deb);
    // bulge._debug_set_H(H_deb);

    // bulge.transform_basis(container);
    // container.to_cpu();
    bulge.execute(container);

    mat_f_N.write_matrix("V1.dat", container.ref_V() );
    mat_f_m.write_matrix("H1.dat", container.ref_V() );
    vec_ops_N.debug_view(container.ref_f(), "f1.dat");

    vec_ops_N.stop_use_vector(f_deb); vec_ops_N.free_vector(f_deb);
    mat_ops_m.stop_use_matrix(Q_deb); mat_ops_m.free_matrix(Q_deb);
    mat_ops_m.stop_use_matrix(H_deb); mat_ops_m.free_matrix(H_deb);




    return 0;
}