//some test for all implemented matrix vector operations;

#include <cmath>
#include <limits>
#include <iostream>
#include <cstdio>
#include <thrust/complex.h>
#include <utils/cuda_support.h>
#include <external_libraries/cublas_wrap.h>
#include <common/gpu_vector_operations.h>
#include <common/gpu_matrix_vector_operations.h>
#include <common/gpu_file_operations.h>
#include <common/gpu_matrix_file_operations.h>
#include <external_libraries/lapack_wrap.h>


//TODO: make a better test case
int main(int argc, char const *argv[])
{
    int N=100;
    int m = 10;
    int k = 4;
    typedef SCALAR_TYPE real;
    typedef thrust::complex<real> complex;

    typedef gpu_vector_operations<real> vec_ops_real_t;
    typedef typename vec_ops_real_t::vector_type vector_type;
    typedef gpu_matrix_vector_operations<real, vector_type> mat_ops_real_t;
    typedef typename mat_ops_real_t::matrix_type matrix_type;
    typedef gpu_file_operations<vec_ops_real_t> files_vec_t;
    typedef gpu_matrix_file_operations<mat_ops_real_t> files_mat_t;

    cublas_wrap *CUBLAS = new cublas_wrap(true);
    vec_ops_real_t *vec_ops_real = new vec_ops_real_t(N, CUBLAS);
    vec_ops_real_t *vec_ops_real_small = new vec_ops_real_t(m, CUBLAS);
    mat_ops_real_t *mat_ops_real = new mat_ops_real_t(N, m, CUBLAS);
    files_vec_t *files_vec_small = new files_vec_t(vec_ops_real_small);    
    mat_ops_real_t *mat_ops_real_small = new mat_ops_real_t(m, k, CUBLAS);
    files_vec_t *files_vec = new files_vec_t(vec_ops_real);
    files_mat_t *files_mat = new files_mat_t(mat_ops_real);
    files_mat_t *files_mat_small = new files_mat_t(mat_ops_real_small);

    matrix_type A;
    matrix_type B;
    matrix_type C;

    vector_type x;
    vector_type y;
    vector_type z;

    vec_ops_real_small->init_vector(y); vec_ops_real_small->start_use_vector(y);
    vec_ops_real_small->init_vector(z); vec_ops_real_small->start_use_vector(z);
    vec_ops_real->init_vector(x); vec_ops_real->start_use_vector(x);
    
    mat_ops_real->init_matrix(A); mat_ops_real->start_use_matrix(A);
    mat_ops_real->init_matrix(C); mat_ops_real->start_use_matrix(C);
    mat_ops_real_small->init_matrix(B); mat_ops_real_small->start_use_matrix(B);

    for(int j=0;j<m;j++)
    {
        vec_ops_real->assign_scalar(real(j), x);
        mat_ops_real->set_matrix_column(A, x, j);

    }

    
    for(int j=0;j<k;j++)
    {
        vec_ops_real_small->assign_random(z); 
        mat_ops_real_small->set_matrix_column(B, z, j);
    }
    
    vec_ops_real->assign_scalar(real(1), x);
    vec_ops_real_small->assign_scalar(real(1), z);    
    mat_ops_real->mat2column_dot_vec(A, m, 1.0, x, 0.0, y);
    mat_ops_real->mat2column_mult_vec(A, m, 1.0, z, 0.0, x);
    
    mat_ops_real->mat2column_mult_mat(A, B, k, 1.0, 0.0, C);


    files_mat->write_matrix("A.dat", A);
    files_mat_small->write_matrix("B.dat", B);
    files_mat->write_matrix("C.dat", C);    
    files_vec->write_vector("x.dat", x);
    files_vec_small->write_vector("z.dat", z);
    files_vec_small->write_vector("y.dat", y);

    mat_ops_real->set_matrix_value(C, 12345.0, 0, 0);
    mat_ops_real->set_matrix_value(C, -12345.0, N-1, m-1);

    files_mat->write_matrix("C_modified.dat", C);

    vec_ops_real_small->stop_use_vector(y); vec_ops_real_small->free_vector(y); 
    vec_ops_real_small->stop_use_vector(z); vec_ops_real_small->free_vector(z);
    vec_ops_real->stop_use_vector(x); vec_ops_real->free_vector(x); 
    mat_ops_real->stop_use_matrix(A); mat_ops_real->free_matrix(A); 
    mat_ops_real_small->stop_use_matrix(B); mat_ops_real_small->free_matrix(B);
    mat_ops_real->stop_use_matrix(C); mat_ops_real->free_matrix(C);

    m = 4;
    
    typedef lapack_wrap<real> lapack_t;
    lapack_t *lapack = new lapack_t(m);

    vector_type Ht = host_allocate<real>(m*m);
    vector_type eig_real = host_allocate<real>(m);
    vector_type eig_imag = host_allocate<real>(m);

    Ht[I2_R(0,0,m)]=0.350000000000000; Ht[I2_R(0,1,m)]=-0.116000000000000;  Ht[I2_R(0,2,m)]=-0.388600000000000; Ht[I2_R(0,3,m)]=-0.294200000000000;
    Ht[I2_R(1,0,m)]=-0.514000000000000; Ht[I2_R(1,1,m)]=0.122500000000000;  Ht[I2_R(1,2,m)]=0.100400000000000; Ht[I2_R(1,3,m)]=0.112600000000000;
    Ht[I2_R(2,0,m)]=0; Ht[I2_R(2,1,m)]=0.644300000000000;  Ht[I2_R(2,2,m)]=-0.135700000000000; Ht[I2_R(2,3,m)]=-0.0977000000000000;
    Ht[I2_R(3,0,m)]=0; Ht[I2_R(3,1,m)]=0;  Ht[I2_R(3,2,m)]=0.426200000000000; Ht[I2_R(3,3,m)]=0.163200000000000;

    real* H_device = device_allocate<real>(m*m);
    host_2_device_cpy<real>(H_device, Ht, m*m);

    //lapack->hessinberg_eigs(Ht, m, eig_real, eig_imag);
    lapack->hessinberg_eigs_from_gpu(H_device, m, eig_real, eig_imag);

    std::cout << "eigenvalues: " << std::endl;
    for(int j=0;j<m;j++)
    {
        real lR = eig_real[j];
        real lI = eig_imag[j];
        if(lI>=0.0)
            std::cout << lR << "+" << lI << "i" << std::endl;
        else
            std::cout << lR << "" << lI << "i" << std::endl;
    }
    device_deallocate<real>(H_device);
    host_deallocate<real>(Ht);
    host_deallocate<real>(eig_real);
    host_deallocate<real>(eig_imag);

    delete lapack;
    delete files_mat;
    delete files_vec;
    delete files_vec_small;
    delete mat_ops_real;
    delete vec_ops_real_small;
    delete vec_ops_real;
    delete CUBLAS;
    return 0;
}
