#include <iostream>
#include <external_libraries/cublas_wrap.h>
#include <common/gpu_vector_operations.h>
#include <utils/pointer_queue.h>
#include <utils/queue_fixed_size.h>



int main(int argc, char const *argv[])
{
    init_cuda(-1);
    
    size_t N = 40*1024*1024;
    
    typedef double T;
    typedef gpu_vector_operations<T> vec_ops_t;
    typedef typename vec_ops_t::vector_type T_vec;


    cublas_wrap CUBLAS;
    vec_ops_t vec_ops(N, (cublas_wrap*)&CUBLAS);

    
    utils::pointer_queue<T> queue(N, 2);
    utils::queue_fixed_size<T, 2> queue_param;

    T_vec x;
    vec_ops.init_vector(x); vec_ops.start_use_vector(x);
    // int aaa = 0;
    for(int j=0;j<500;j++)
    {
        vec_ops.assign_scalar(T(j), x);
        queue.push(x);
        queue_param.push(j);
        T norm_x = vec_ops.norm(x);

        T norm_0 = vec_ops.norm(queue.at(0));
        T id0 = queue_param.at(0);
        T norm_1 = -1;
        T id1  =-1;
        if( queue.is_queue_filled() )
        {
            id1 = queue_param.at(1);
            norm_1 = vec_ops.norm(queue.at(1));
        }

        std::cout << "norm(x) = " << norm_x << " id0 = " << id0 <<  " norm_0 = " << norm_0 << " id1 = " << id1 << " norm_1 = " << norm_1 << std::endl;
        
    }

    T norm_x = vec_ops.norm(x);
    T norm_vec1 = vec_ops.norm(queue.at(1));

    std::cout << "norm_1 = " << norm_x << " norm_2 = " << norm_vec1 << " norm difference = " << ( norm_vec1 - norm_x ) << std::endl;

   

    vec_ops.stop_use_vector(x); vec_ops.free_vector(x);
    
    return 0;
}