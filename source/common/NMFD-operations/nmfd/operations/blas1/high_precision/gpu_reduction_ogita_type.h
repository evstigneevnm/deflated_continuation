#ifndef __GPU_REDUCTION_OGITA_TYPE_H__
#define __GPU_REDUCTION_OGITA_TYPE_H__

#include <thrust/complex.h>

namespace gpu_reduction_ogita_type{

template<typename T_>
struct type_complex_cast
{
    using T = T_;
};
  
template<>
struct type_complex_cast< thrust::complex<float> >
{
    using T = float;
};
template<>
struct type_complex_cast< thrust::complex<double> >
{
    using T = double;
};    


template<typename T>
struct return_real
{
    using T_real = typename type_complex_cast<T>::T;
    T_real get_real(T val)
    {
        return val;
    }    
};


template<>
struct return_real< thrust::complex<float> >
{
    using T_real = float;//typename type_complex_cast< thrust::complex<float> >::T;
    T_real get_real(thrust::complex<float> val)
    {
        return val.real();
    }    
};
template<>
struct return_real< thrust::complex<double> >
{
    using T_real = double;//typename type_complex_cast< thrust::complex<double> >::T;
    T_real get_real(thrust::complex<double> val)
    {
        return val.real();
    }    
};

}


#endif

    