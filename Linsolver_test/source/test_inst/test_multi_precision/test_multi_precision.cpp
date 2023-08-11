#include <boost/multiprecision/cpp_bin_float.hpp>
#include <iostream>

int main() {
    boost::multiprecision::cpp_bin_float_100 a = 1;
    boost::multiprecision::cpp_bin_float_100 b = 10;

    using cpp_bin_float_200 = boost::multiprecision::number<boost::multiprecision::backends::cpp_bin_float<200> >;

    cpp_bin_float_200 a200 = 1;

    std::cout << std::setprecision(50) << std::endl;

    std::cout << boost::multiprecision::sin(a200) << std::endl;
    std::cout << boost::multiprecision::sinh(a200) << std::endl;
    std::cout << boost::multiprecision::cos(a200) << std::endl;
    std::cout << boost::multiprecision::cosh(a200) << std::endl;
    std::cout << boost::multiprecision::tan(a200) << std::endl;
    std::cout << boost::multiprecision::exp(a200) << std::endl;
    std::cout << boost::multiprecision::log(b) << std::endl;
    std::cout << boost::multiprecision::sqrt(b) << std::endl;

    std::cout << "cmp float_100 and float_200" << std::endl; 
    std::cout << std::setprecision(120) << std::endl;
    std::cout << boost::multiprecision::sinh(a) << std::endl;
    std::cout << boost::multiprecision::sinh(a200) << std::endl;
    std::cout << std::setprecision(17) << std::endl;
    std::cout << "diff = " << boost::multiprecision::sinh(a200) - boost::multiprecision::sinh(a) << std::endl;

    //impement tests for vector and matrix operations

    return 0;
}