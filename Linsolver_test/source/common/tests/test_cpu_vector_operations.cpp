#include <iostream>
#include <string>
#include <common/cpu_vector_operations.h>




int main(int argc, char const *argv[]) 
{
    
    int N=10;//25600;
    typedef SCALAR_TYPE real;
    using vec_ops_t = cpu_vector_operations<real>;
    using vec_t = typename vec_ops_t::vector_type;


    vec_ops_t vec_ops(N);
    std::cout << "testing slices: " << std::endl;

    vec_t x,y,z;
    vec_ops.init_vectors(x,y,z); vec_ops.start_use_vector(x, 10); vec_ops.start_use_vector(y, 3); vec_ops.start_use_vector(z, 9);
    std::cout << "x.size() = " << x.size() << std::endl;
    std::cout << "y.size() = " << y.size() << std::endl;
    for(int j = 0;j<10;j++)
    {
        x[j] = j;
    }
    std::cout << "x:";
    for(auto& x_l: x)
    {
        std::cout << x_l << " ";
    }
    std::cout << std::endl;

    vec_ops.assign_slices(x, {{3,4},{6,8}}, y);
    vec_ops.assign_skip_slices(x, {{5,6}}, z);

    std::cout << "y:";
    for(auto& y_l: y)
    {
        std::cout << y_l << " ";
    }
    std::cout << std::endl;    
    std::cout << "z:";    
    for(auto& z_l: z)
    {
        std::cout << z_l << " ";
    }
    std::cout << std::endl;
    vec_ops.stop_use_vectors(x,y,z); vec_ops.free_vectors(x,y,z);


    return 0;
}