// Example program
#include <iostream>
#include <vector>
#include <common/cpu_vector_operations.h>
#include "curve_container.h"




int main()
{
    typedef SFLOATTYPE real;
    typedef knots<real> knots_t;
    typedef cpu_vector_operations<real> cpu_vec_ops_t;
    typedef typename cpu_vec_ops_t::vector_type real_vec;
    //typedef curve_storage<cpu_vec_ops_t, knots_t> curve_storage_t;

    size_t vector_size = 10;

    knots_t knots_test;
    cpu_vec_ops_t cpu_vec_ops(vector_size);
    
    // curve_storage_t curve_storage(&cpu_vec_ops, &knots_test);

    //testing knots
    knots_test.add_element(real(2));
    knots_test.add_element({1, 2.5, 5, 4, 1, 1});
    knots_test.add_element({1, 2.5, 3-0.5 ,4, 5, 5, 6, 6, 6, 1});
    knots_test.add_element(real(2));
    knots_test.add_element(2);
    knots_test.add_element(3);
    knots_test.add_element(3);
    knots_test.add_element(3);
    knots_test.add_element(6.5);
    knots_test.add_element({2, 5, 4.0, 3, 3});
    knots_test.add_element({0.5, 0.4, 0.4, 0.5, 1});
    knots_test.add_element(0.5);

    std::cout << "knots: ";

    for(auto &x: knots_test)
    {
        std::cout << x << " ";
    }
    std::cout << std::endl;
    
    std::cout << "there are " << knots_test.size() << " keys\n";
    
    real_vec x;
    cpu_vec_ops.init_vector(x); cpu_vec_ops.start_use_vector(x); 
    real lambda = 3.4;

    // curve_storage.set_knots();
    
    // curve_storage.add(lambda, x);


    cpu_vec_ops.stop_use_vector(x); cpu_vec_ops.free_vector(x); 
    return 0;
}