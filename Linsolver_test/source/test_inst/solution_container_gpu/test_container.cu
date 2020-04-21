#include <cmath>
#include <iostream>
#include <cstdio>

#include <utils/cuda_support.h>
#include <utils/log.h>
#include <external_libraries/cufft_wrap.h>
#include <external_libraries/cublas_wrap.h>

#include <deflation/solution_storage.h>

#include <common/gpu_vector_operations.h>
#include <common/gpu_file_operations.h>


int main(int argc, char const *argv[])
{
   
    typedef double T;
    typedef T* T_vec;
    typedef gpu_vector_operations<T> vec_ops_real;
    typedef gpu_file_operations<vec_ops_real> file_ops_t;
    typedef deflation::solution_storage<vec_ops_real> sol_storage_def_t;

    init_cuda(5);
    size_t Nx=2*1024;
    size_t Ny=2*1024;
    T norm_wight = std::sqrt(T(Nx*Ny));
    cublas_wrap *CUBLAS = new cublas_wrap();
    CUBLAS->set_pointer_location_device(false);
    vec_ops_real *vec_ops = new vec_ops_real(Nx*Ny, CUBLAS);
    file_ops_t *file_ops = new file_ops_t(vec_ops);

    sol_storage_def_t *sol_storage = new sol_storage_def_t(vec_ops, 100, norm_wight);
    sol_storage_def_t sol_storage_obj(vec_ops, 10, norm_wight);
    T_vec x0, x1, x1o;
    int nnn;
    std::cout << "start here";
    std::cin  >> nnn;
    vec_ops->init_vector(x0); vec_ops->start_use_vector(x0);    
    //vec_ops->init_vector(x1); vec_ops->start_use_vector(x1);    
    //vec_ops->init_vector(x1o); vec_ops->start_use_vector(x1o);    
    vec_ops->assign_scalar(T(1), x0);

    std::cout << "allocated and assigned vector\n";
    std::cout << "enter how many vectors to push>>>";
    std::cin  >> nnn;
    for(int jj=0;jj<nnn;jj++)
    {
        sol_storage->push_back(x0);
        std::cout << "pushed " << jj+1 << "-th vector \n";
    }
    //(*sol_storage)[0].copy(x1);
    
    //sol_storage_obj.push(x0);
    //x1o = sol_storage_obj[0].get_ref();
    //sol_storage_obj[0].copy(x1o);
    std::cout << "current size of container:";
    std::cout << sol_storage->get_size() << std::endl;
    std::cin  >> nnn;

    vec_ops->stop_use_vector(x0); vec_ops->free_vector(x0);
   // vec_ops->stop_use_vector(x1); vec_ops->free_vector(x1);
   // vec_ops->stop_use_vector(x1o); vec_ops->free_vector(x1o);
    std::cout << "removed vector";
    std::cin  >> nnn;

    sol_storage->clear();

    std::cout << "clear(). Current size of container:";
    std::cout << sol_storage->get_size() << std::endl;
    
    std::cin  >> nnn;


    delete sol_storage;
    delete file_ops;
    delete vec_ops;
    delete CUBLAS;

    return 0;
}