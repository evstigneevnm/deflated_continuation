#include <iostream>
#include "json_test.h"



int main(int argc, char const *argv[])
{
    parameters<double> p = read_json<double>("data.json");


    std::cout << "a_e = " << p.a_e << std::endl;
    std::cout << "b_e = " << p.b_e << std::endl;
    std::cout << "a_i = " << p.internal.a_i << std::endl;
    std::cout << "b_i = " << p.internal.b_i << std::endl;
    std::cout << "a_i_i = " << p.internal.internal.a_2 << std::endl;
    std::cout << "b_i_i = " << p.internal.internal.b_2 << std::endl;
    std::cout << "keys = " << std::endl;
    for(auto &x: p.internal.internal.keys)
    {
        std::cout << x << " ";
    }
    std::cout << std::endl;
    return 0;
}