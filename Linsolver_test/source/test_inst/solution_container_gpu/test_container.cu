#include <cmath>
#include <iostream>
#include <cstdio>

#include <utils/cuda_support.h>
#include <utils/log.h>
#include <external_libraries/cufft_wrap.h>
#include <external_libraries/cublas_wrap.h>

#include <deflation/solution_storage.h>

#include <gpu_vector_operations.h>
#include "gpu_file_operations.h"


int main(int argc, char const *argv[])
{
   
    typedef double T;
    typedef T* T_vec;
    typedef gpu_vector_operations<T> vec_ops_real;
    typedef deflation::solution_storage<vec_ops_real> sol_storage_def_t;

    init_cuda(5);
    size_t Nx=128;
    size_t Ny=128;
    T norm_wight = std::sqrt(T(Nx*Ny));
    cublas_wrap *CUBLAS = new cublas_wrap();
    CUBLAS->set_pointer_location_device(false);
        
    vec_ops_real *vec_ops = new vec_ops_real(Nx*Ny, CUBLAS);
    sol_storage_def_t *sol_storage = new sol_storage_def_t(vec_ops, 10, norm_wight);
    sol_storage_def_t sol_storage_obj(vec_ops, 10, norm_wight);
    T_vec x0, x1, x1o;
    vec_ops->init_vector(x0); vec_ops->start_use_vector(x0);    
    vec_ops->init_vector(x1); vec_ops->start_use_vector(x1);    
    vec_ops->init_vector(x1o); vec_ops->start_use_vector(x1o);    

    vec_ops->assign_scalar(T(1), x0);
    sol_storage->push(x0);
    (*sol_storage)[0].copy(x1);
    
    sol_storage_obj.push(x0);
    //x1o = sol_storage_obj[0].get_ref();
    sol_storage_obj[0].copy(x1o);

    //vec_ops->assign(x0,x1);
    //vec_ops->assign(x0,x1o);
    //CUBLAS->copy<T>(Nx*Ny, x0, x1);
    gpu_file_operations::write_matrix<T>("x0_out.dat", Nx, Ny, x0);
    gpu_file_operations::write_matrix<T>("x1_out.dat", Nx, Ny, x1);
    gpu_file_operations::write_matrix<T>("x1o_out.dat", Nx, Ny, x1o);

    vec_ops->stop_use_vector(x0); vec_ops->free_vector(x0);
    vec_ops->stop_use_vector(x1); vec_ops->free_vector(x1);
    vec_ops->stop_use_vector(x1o); vec_ops->free_vector(x1o);
    

    delete sol_storage;
    delete vec_ops;
    delete CUBLAS;

    return 0;
}