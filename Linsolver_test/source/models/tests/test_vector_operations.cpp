//some test for all implemented vector operations;

#include <cmath>
#include <limits>
#include <iostream>
#include <cstdio>
#include <thrust/complex.h>
#include <utils/cuda_support.h>
#include <external_libraries/cufft_wrap.h>
#include <external_libraries/cublas_wrap.h>
#include <common/gpu_vector_operations.h>
#include <common/gpu_file_operations.h>



int main(int argc, char const *argv[])
{
    int N=10;//25600;
    typedef SCALAR_TYPE real;
    typedef thrust::complex<real> complex;
    typedef gpu_vector_operations<real> gpu_vector_operations_real;
    typedef gpu_vector_operations<complex> gpu_vector_operations_complex;

    typedef gpu_file_operations<gpu_vector_operations_real> files_real_t;
    typedef gpu_file_operations<gpu_vector_operations_complex> files_complex_t;
    init_cuda(5);

    cublas_wrap *CUBLAS = new cublas_wrap(true);
    gpu_vector_operations_real *vec_ops_R = new gpu_vector_operations_real(N, CUBLAS);
    gpu_vector_operations_complex *vec_ops_C = new gpu_vector_operations_complex(N, CUBLAS);
    files_real_t *files_R = new files_real_t(vec_ops_R);
    files_complex_t *files_C = new files_complex_t(vec_ops_C);


    real *u1_d, *u2_d, *u3_d;
    complex *z1_d, *z2_d, *z3_d;
    vec_ops_R->init_vector(u1_d); vec_ops_R->init_vector(u2_d); vec_ops_R->init_vector(u3_d);
    vec_ops_C->init_vector(z1_d); vec_ops_C->init_vector(z2_d); vec_ops_C->init_vector(z3_d);
    vec_ops_R->start_use_vector(u1_d); vec_ops_R->start_use_vector(u2_d); vec_ops_R->start_use_vector(u3_d);
    vec_ops_C->start_use_vector(z1_d); vec_ops_C->start_use_vector(z2_d); vec_ops_C->start_use_vector(z3_d);

    //vec_ops_R->assign_scalar(0.5, u1_d);
    printf("using vectors of size = %i\n", N);
    vec_ops_R->assign_random(u1_d);
    real norm_u1 = vec_ops_R->norm(u1_d);
    printf("||u1||=%lf\n", double(norm_u1));
    auto max_u1 = vec_ops_R->max_argmax_element(u1_d);
    printf("max(u1) = %lf, argmax(u1) = %d\n", max_u1.first, max_u1.second);

    vec_ops_C->assign_scalar(complex(1.0), z1_d);

    vec_ops_R->assign_scalar(-0.5, u2_d); 
    vec_ops_C->assign_scalar(complex(-0.5,0.3), z2_d);

    vec_ops_R->assign_scalar(123.0, u3_d); 
    vec_ops_C->assign_scalar(complex(0,2.2), z3_d);

    vec_ops_R->mul_pointwise(1.0, u1_d, 0.2, u2_d, 
                        u3_d);
    
    vec_ops_C->add_mul(complex(0,1), z1_d, complex(2.0), (complex*&) u2_d, complex(-1.0), z3_d);

    vec_ops_R->add_mul_scalar(1.0, 0.5, u3_d);
    vec_ops_C->add_mul_scalar(complex(0,1.0), complex(0.5,0.5), z3_d);

    vec_ops_C->swap(z3_d,z2_d);
    vec_ops_R->assign_mul(2.0, u1_d, u3_d);
    vec_ops_C->add_mul(complex(0.0,2.3), z1_d, z3_d);
    //vec_ops_C->assign_mul(1.0, z3_d, 2.0, z2_d, z1_d);
    real z1norm=0.0;
    vec_ops_C->norm(z1_d, &z1norm);
    printf("||z1||=%lf\n", double(z1norm));

    real norm_rank1_test = vec_ops_C->norm_rank1(z1_d, complex(0,1.0) );
    printf("|| [z1; 0+1i] || = %lf\n", double(norm_rank1_test));

    vec_ops_R->add_mul(2.0, u1_d, 2.0, u3_d);

    printf("real vector norm_{inf} = %lf, should be 0.5\n", (double) vec_ops_R->norm_inf(u2_d) );

    printf("result real vector norm2 =%le\n",(double) vec_ops_R->normalize(u3_d));
    printf("after normalization=%le\n",(double) vec_ops_R->norm(u3_d));

    complex scal_prod_test = sqrt(vec_ops_C->scalar_prod(z3_d,z3_d));
    printf("result for complex vector dot prod (z,z)^1/2=(%le,%le), check no imag part!\n", (double) scal_prod_test.real(), (double) scal_prod_test.imag() );
    printf("result complex vector norm=%le\n",(double) vec_ops_C->normalize(z3_d));
    printf("after normalization=%le\n",(double) vec_ops_C->norm(z3_d));

    printf("populating a single nan in real v3 vector and a single inf into complex v2 vector!\n");
    
    complex *z2_h=(complex*)malloc(N*sizeof(complex));
    real *u3_h=(real*)malloc(N*sizeof(real));
    device_2_host_cpy<real>(u3_h, u3_d, N);
    device_2_host_cpy<complex>(z2_h, z2_d, N);
    u3_h[N-1]=std::numeric_limits<real>::quiet_NaN();
    z2_h[N-1]=std::numeric_limits<real>::infinity();
    host_2_device_cpy<real>(u3_d, u3_h, N);
    host_2_device_cpy<complex>(z2_d, z2_h, N);

    printf("is result finite?:\n");
    printf("        v1, v2, v3\n");
    printf("R vecs: %s, %s, %s\n",true==vec_ops_R->check_is_valid_number(u1_d)?"T":"F",true==vec_ops_R->check_is_valid_number(u2_d)?"T":"F",true==vec_ops_R->check_is_valid_number(u3_d)?"T":"F");
    printf("C vecs: %s, %s, %s\n",true==vec_ops_C->check_is_valid_number(z1_d)?"T":"F",true==vec_ops_C->check_is_valid_number(z2_d)?"T":"F",true==vec_ops_C->check_is_valid_number(z3_d)?"T":"F");

    // files_R->write_vector("real_vector_test.dat", u3_d);
    files_C->write_vector("complex_vector_test.dat", z2_d);    

    files_R->read_vector("real_vector_test0.dat", u3_d);
    files_R->write_vector("real_vector_test.dat", u3_d);

    vec_ops_R->free_vector(u1_d); vec_ops_R->free_vector(u2_d); vec_ops_R->free_vector(u3_d); 
    vec_ops_C->free_vector(z1_d); vec_ops_C->free_vector(z2_d); vec_ops_C->free_vector(z3_d);

    delete files_C;
    delete files_R;
    delete vec_ops_C;
    delete vec_ops_R;
    delete CUBLAS;
    return 0;
}