#include <iostream>
#include "solutions_container.h"
#include <common/cpu_vector_operations.h>

typedef SFLOATTYPE real;
typedef cpu_vector_operations<real> vector_operations;
typedef typename vector_operations::vector_type  vector_t;

void init_vec(size_t N, vector_t& array, real init)
{
    //test CPU only
    for(int j=0;j<N;j++)
        array[j]=real(j)+init;

}



int main(int argc, char const *argv[])
{
    size_t N = 500000;
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
      
    init_vec(N, array, 0.1);    
    ST.push( array);  

    init_vec(N, array, 5.5);
    ST.push( array);

    ST[0][4]=7.777;
    
    for(int j=0;j<ST.get_size();j++)
    {
        std::cout << ST[j][4] << std::endl;
    }
    std::cout << "sizeof(solution_storage) = " << sizeof(solution_storage<vector_operations>) << std::endl;
    // can check if public
    //std::cout << "sizeof(internal_container) = " << sizeof(solution_storage<vector_operations>::internal_container) << std::endl;
    
    real beta=-1;
    vector_t c;
    vec_ops->init_vector(c); vec_ops->start_use_vector(c);

    init_vec(N, array, 5.5005);

    ST.calc_distance(array, beta, c, 2);
    printf("\n beta-1 = %le, norm(c) = %le\n", double(beta-1), double(vec_ops->norm(c)) );

    vec_ops->free_vector(c); vec_ops->stop_use_vector(c);
    vec_ops->free_vector(array); vec_ops->stop_use_vector(array);
    delete [] vec_ops; 
    return 0;
}

