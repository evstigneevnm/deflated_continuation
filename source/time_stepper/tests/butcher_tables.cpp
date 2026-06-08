#include <iostream>
#include <time_stepper/detail/all_methods_enum.h>
#include <time_stepper/detail/butcher_tables.h>


int main(int argc, char const *argv[])
{

    time_steppers::detail::butcher_tables tables;
    time_steppers::detail::composite_butcher_tables composite_tables;

    using table_t = time_steppers::detail::tableu;
    using composite_tableu_t = time_steppers::detail::composite_tableu;

    auto table_EULER = tables.set_table(time_steppers::detail::methods::EXPLICIT_EULER);
    auto table_HEUN_EULER = tables.set_table(time_steppers::detail::methods::HEUN_EULER);
    auto table_RKDP45 = tables.set_table(time_steppers::detail::methods::RKDP45);
    auto table_RK33SSP = tables.set_table(time_steppers::detail::methods::RK33SSP);
    auto table_RK43SSP = tables.set_table(time_steppers::detail::methods::RK43SSP);
    auto table_RK64SSP = tables.set_table(time_steppers::detail::methods::RK64SSP);
    
    auto table_IMEX_EULER = composite_tables.set_table(time_steppers::detail::methods::IMEX_EULER);
    auto table_IMEX_TR2 = composite_tables.set_table(time_steppers::detail::methods::IMEX_TR2);

    std::cout << "table names: ";
    for(auto &name: tables.get_list_of_table_names() )
    {
        std::cout << name << " ";
    }
    std::cout << std::endl;


    std::cout << "composite table names: ";
    for(auto &name: composite_tables.get_list_of_table_names() )
    {
        std::cout << name << " ";
    }
    std::cout << std::endl;

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

    auto print_mat_p = [&print_mat](const std::pair<table_t, table_t>& table_p)
    {
        std::cout << "explicit: " << std::endl;
        print_mat(table_p.first);
        std::cout << "implicit: " << std::endl;
        print_mat(table_p.second);
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
    auto print_vecs_p = [&print_vecs](const std::pair<table_t, table_t>& table_p)
    {
        std::cout << "explicit: " << std::endl;
        print_vecs(table_p.first);
        std::cout << "implicit: " << std::endl;        
        print_vecs(table_p.second);
    };


//     std::cout << "Explicit Euler: " << std::endl;
//     print_mat(table_EULER);
//     print_vecs(table_EULER);

//     std::cout << "Heun Euler: " << std::endl;
//     print_mat(table_HEUN_EULER);
//     print_vecs(table_HEUN_EULER);

//     std::cout << "RKDP45: " << std::endl;
//     print_mat(table_RKDP45);
//     print_vecs(table_RKDP45);

//     std::cout << "RK33SSP: " << std::endl;
//     print_mat(table_RK33SSP);
//     print_vecs(table_RK33SSP);   

//     std::cout << "RK43SSP: " << std::endl;
//     print_mat(table_RK43SSP);
//     print_vecs(table_RK43SSP);     

//     std::cout << "RK64SSP: " << std::endl;
//     print_mat(table_RK64SSP);
//     print_vecs(table_RK64SSP);       
// //  to be continued...

//     std::cout << "IMEX_EULER: " << std::endl;
//     print_mat_p(table_IMEX_EULER);
//     print_vecs_p(table_IMEX_EULER);
    
//     std::cout << "IMEX_TR2: " << std::endl;
    // print_mat_p(table_IMEX_TR2);
    // print_vecs_p(table_IMEX_TR2);
    
    std::cout << "test by name:" << std::endl;
    auto names = tables.get_list_of_table_names();
    for(auto &n: names)
    {
        std::cout << n << ": " << std::endl;
        auto tt = tables.set_table_by_name(n);
        print_mat(tt);
        print_vecs(tt);    
        std::cout << "=================" << std::endl;
    }



    return 0;
}