#include <iostream>
#include <time_stepper/detail/all_methods_enum.h>
#include <time_stepper/detail/butcher_tables.h>


int main(int argc, char const *argv[])
{

    time_steppers::detail::butcher_tables tables;
    using table_t = time_steppers::detail::tableu;

    auto table_EULER = tables.set_table(time_steppers::detail::methods::EXPLICIT_EULER);
    auto table_HEUN_EULER = tables.set_table(time_steppers::detail::methods::HEUN_EULER);
    auto table_RKDP45 = tables.set_table(time_steppers::detail::methods::RKDP45);
    auto table_RK33SSP = tables.set_table(time_steppers::detail::methods::RK33SSP);
    auto table_RK43SSP = tables.set_table(time_steppers::detail::methods::RK43SSP);
    auto table_RK64SSP = tables.set_table(time_steppers::detail::methods::RK64SSP);
    
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
    print_mat(table_EULER);
    print_vecs(table_EULER);

    std::cout << "Heun Euler: " << std::endl;
    print_mat(table_HEUN_EULER);
    print_vecs(table_HEUN_EULER);

    std::cout << "RKDP45: " << std::endl;
    print_mat(table_RKDP45);
    print_vecs(table_RKDP45);

    std::cout << "RK33SSP: " << std::endl;
    print_mat(table_RK33SSP);
    print_vecs(table_RK33SSP);   

    std::cout << "RK43SSP: " << std::endl;
    print_mat(table_RK43SSP);
    print_vecs(table_RK43SSP);     

    std::cout << "RK64SSP: " << std::endl;
    print_mat(table_RK64SSP);
    print_vecs(table_RK64SSP);       
//  to be continued...

    
    return 0;
}