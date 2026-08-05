#ifndef __DISCRETIZATION_FOURIER_DIAGONAL_SYMBOLS_H__
#define __DISCRETIZATION_FOURIER_DIAGONAL_SYMBOLS_H__

namespace discretization
{
namespace fourier
{

template<class T>
struct diagonal_symbols
{
    static T laplacian(const T k_squared)
    {
        return -k_squared;
    }

    static T biharmonic(const T k_squared)
    {
        return k_squared*k_squared;
    }

    static T inverse_laplacian(const T k_squared)
    {
        return k_squared == T(0) ? T(0) : -T(1)/k_squared;
    }
};

} // namespace fourier
} // namespace discretization

#endif
