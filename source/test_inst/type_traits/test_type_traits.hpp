#pragma once
#include <iostream>

template<class Vector>
struct nonlinear_operator_test_1
{
    struct is_periodic_orbit_reprojected
    {
        static const bool value = false;
    };

    void F(Vector* vec)
    {
        std::cout << "op1.F" << std::endl;
    }

};
template<class Vector>
struct nonlinear_operator_test_2
{
    struct is_periodic_orbit_reprojected
    {
        static const bool value = true;
    };

    void F(Vector* vec)
    {
        std::cout << "op2.F" << std::endl;
    }

    void reproject(Vector* vec)
    {
        std::cout << "op2.reproject" << std::endl;
    }

};