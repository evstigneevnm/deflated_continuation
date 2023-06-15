#include <iostream>
#include <time_stepper/detail/all_methods_enum.h>
#include <time_stepper/detail/butcher_tables.h>


int main(int argc, char const *argv[])
{

    time_steppers::detail::butcher_tables tables;
    using table_t = time_steppers::detail::tableu;

    auto table_Euler = tables.set_table(time_steppers::detail::methods::EXPLICIT_EULER);
    auto table_RK45 = tables.set_table(time_steppers::detail::methods::RKDP45);

    auto print_mat = [](const table_t& table_p)
    {
        size_t sz = table_p.get_size();
        std::cout << "A: " << std::endl;
        for(size_t j=0;j<sz;j++)
        {
            for(size_t k=0;k<sz;k++)
            {
                std::cout << table_p.get_A<long double>(j,k) << " ";
            }
            std::cout << std::endl;
        }
        
    };
    auto print_vecs = [](const table_t& table_p)
    {
        size_t sz = table_p.get_size();
        std::cout << "b: " << std::endl;
        for(size_t j=0;j<sz;j++)
        {
            std::cout << table_p.get_b<long double>(j) << " ";
        }
        std::cout << std::endl;
        if(!table_p.is_autonomous())
        {
            std::cout << "c: " << std::endl;
            for(size_t j=0;j<sz;j++)
            {
                std::cout << table_p.get_c<long double>(j) << " ";
            }
            std::cout << std::endl;            
        }
        if(table_p.is_embedded())
        {
            std::cout << "b-b_hat: " << std::endl;
            for(size_t j=0;j<sz;j++)
            {
                std::cout << table_p.get_err_b<long double>(j) << " ";
            }
            std::cout << std::endl;            
        }

    };    
    std::cout << "Explicit Euler: " << std::endl;
    print_mat(table_Euler);
    print_vecs(table_Euler);
    std::cout << "RK45: " << std::endl;
    print_mat(table_RK45);
    print_vecs(table_RK45);
//  to be continued...

    
    return 0;
}