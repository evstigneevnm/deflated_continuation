#include <iostream>
#include "solutions_container.h"
#include <cpu_vector_operations.h>

typedef SFLOATTYPE real;
typedef real* vector_t;
typedef cpu_vector_operations<real> vector_operations;

void init_vec(size_t N, vector_t& array, real init)
{
    for(int j=0;j<N;j++)
        array[j]=j+init;
}



int main(int argc, char const *argv[])
{
    size_t N = 1500000;
    vector_operations *vec_ops = new vector_operations(N);
    solution_storage<vector_operations> ST(vec_ops, 10);
    vector_t array;
    vec_ops->init_vector(array); vec_ops->start_use_vector(array);

    init_vec(N, array, 1);
    ST.push( array);
    
    init_vec(N, array, 11);
    ST.push( array);
    
    init_vec(N, array, 111);    
    ST.push( array);
    
    vec_ops->free_vector(array); vec_ops->stop_use_vector(array);

    ST[0][4]=7.777;
    
    for(int j=0;j<ST.get_size();j++)
    {
        std::cout << ST[j][4] << std::endl;
    }
    
    delete [] vec_ops; 
    return 0;
}

