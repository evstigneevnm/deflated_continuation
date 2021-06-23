#ifndef __GPU_REDUCTION_IMPL_OGITA_FUNCTIONS_CUH__
#define __GPU_REDUCTION_IMPL_OGITA_FUNCTIONS_CUH__

#include <cuda_runtime_api.h>
#include <thrust/complex.h>
#include <common/testing/gpu_reduction_ogita_type.h>

namespace gpu_reduction_ogita_gpu_kernels
{


// shuffle functions for CUDA_ARCH>=3
#if (__CUDA_ARCH__ >= 300 )

//service function
template<class T>
__device__ __forceinline__ T _shfl(T a, const int j)
{
   
    // WarpSize equals 32 (default third param)
    #if(CUDART_VERSION < 9000)
    return __shfl_xor(a, j); //for 8.0<= cuda < 9.0
    #else
    return __shfl_xor_sync(0xFFFFFFFF, a, j); //for cuda >=9 
    // -1 = 0xFFFFFFFF assumes that threadsize = 32, see cuda/include/cooperative_groups.h
    #endif
     
    
}

//basic template (int, float)
template<class T>
__device__ __forceinline__ T shuffle(T a, const int j)
{
    return _shfl<T>(T(a), j);
}
//double specialization
template<>
__device__ __forceinline__ double shuffle(double a, const int j)
{
    //make double out of two ints
    // int a_hi = __double2hiint(a);
    // int a_low = __double2loint(a);
    // int hi = _shfl<int>(a_hi, j);
    // int low = _shfl<int>(a_low, j );
    // return __hiloint2double( hi, low );
    return __hiloint2double( _shfl<int>(__double2hiint(a), j), _shfl<int>(__double2loint(a), j) );
}

//complex<float> specialization
template<>
__device__ __forceinline__ thrust::complex<float> shuffle(thrust::complex<float> a, const int j)
{
    return thrust::complex<float>(shuffle<float>(a.real(), j), shuffle<float>(a.imag(), j) );
}
//complex<double> specialization
template<>
__device__ __forceinline__ thrust::complex<double> shuffle(thrust::complex<double> a, const int j)
{
    return thrust::complex<double>(shuffle<double>(a.real(), j), shuffle<double>(a.imag(), j) );
}

#endif


// FMA specilization
template <class T>
__device__ __forceinline__ T _fma(T a, T b, T c)
{
}
template <>
__device__ __forceinline__ float _fma(float a, float b, float c)
{
    return __fmaf_ieee_rn(a, b, c);
}

template <>
__device__ __forceinline__ double _fma(double a, double b, double c)
{
    return __fma_rn(a, b, c);
}

//this needs specialization for complex, since no fma exist for thrust::complex
template<class T>
__device__ inline T __GPU_REDUCTION_OGITA_H__two_prod_device(T &t, T a, T b)
{
    T p = a*b;
    t = _fma(a, b, -p);
    return p;
}

template<class T>
__device__ __forceinline__ T __GPU_REDUCTION_OGITA_H__two_sum_device(T &t, T a, T b)
{
    T s = a+b;
    T bs = s-a;
    T as = s-bs;
    t = (b-bs) + (a-as);
    return s;
}


template<>
__device__ __forceinline__ thrust::complex<float> __GPU_REDUCTION_OGITA_H__two_prod_device(thrust::complex<float> &t, thrust::complex<float> a, thrust::complex<float> b)
{
    using T_real = float;
    using TC = typename thrust::complex<T_real>;
    // T p = a*b;
    // t = fma(a, b, -p);
    // return p;

    T_real a_R = a.real();
    T_real a_I = a.imag();
    T_real b_R = b.real();
    T_real b_I = b.imag();

    T_real p_R1 = a_R*b_R;
    T_real t_R1 = _fma<T_real>(a_R, b_R, -p_R1);
    T_real p_R2 = a_I*b_I;
    T_real t_R2 = _fma<T_real>(a_I, b_I, -p_R2);
    T_real p_I1 = a_R*b_I;
    T_real t_I1 = _fma<T_real>(a_R, b_I, -p_I1);
    T_real p_I2 = -a_I*b_R;
    T_real t_I2 = _fma<T_real>(-a_I, b_R, -p_I2);

    T_real t1 = T_real(0.0);
    T_real t2 = T_real(0.0);
    T_real p_R = __GPU_REDUCTION_OGITA_H__two_sum_device<T_real>(t1, p_R1, p_R2);
    T_real p_I = __GPU_REDUCTION_OGITA_H__two_sum_device<T_real>(t2, p_I1, p_I2);
    
    TC p = TC(p_R, p_I);
    
    t = TC(t_R1 + t_R2 + t1, t_I1 + t_I2 + t2);
    // printf("p_R1=%e, p_R2=%e, p_R=%e p_I1=%e, p_I2=%e, p_I=%e\n", p_R1, p_R2, p_R, p_I1, p_I2, p_I );
    // printf("t = (%le,%le)\n", (double)t.real(), (double)t.imag());
    return p;    
}
template<>
__device__ __forceinline__ thrust::complex<double> __GPU_REDUCTION_OGITA_H__two_prod_device(thrust::complex<double> &t, thrust::complex<double> a, thrust::complex<double> b)
{
    using T_real = double;
    using TC = typename thrust::complex<T_real>;
    // T p = a*b;
    // t = fma(a, b, -p);
    // return p;

    T_real a_R = a.real();
    T_real a_I = a.imag();
    T_real b_R = b.real();
    T_real b_I = b.imag();

    T_real p_R1 = a_R*b_R;
    T_real t_R1 = _fma<T_real>(a_R, b_R, -p_R1);
    T_real p_R2 = a_I*b_I;
    T_real t_R2 = _fma<T_real>(a_I, b_I, -p_R2);
    T_real p_I1 = a_R*b_I;
    T_real t_I1 = _fma<T_real>(a_R, b_I, -p_I1);
    T_real p_I2 = -a_I*b_R;
    T_real t_I2 = _fma<T_real>(-a_I, b_R, -p_I2);

    T_real t1 = T_real(0.0);
    T_real t2 = T_real(0.0);
    T_real p_R = __GPU_REDUCTION_OGITA_H__two_sum_device<T_real>(t1, p_R1, p_R2);
    T_real p_I = __GPU_REDUCTION_OGITA_H__two_sum_device<T_real>(t2, p_I1, p_I2);
 
    TC p = TC(p_R, p_I);
    
    t = TC(t_R1 + t_R2 + t1, t_I1 + t_I2 + t2);

    // printf("p_R1 = %le, p_R2 = %le\n", p_R1, p_R2 );
    return p;    
}

template<class T>
__device__ __forceinline__ typename gpu_reduction_ogita_type::type_complex_cast<T>::T cuda_abs(T val)
{
    return( abs(val) );
}
template<>
__device__ __forceinline__ double cuda_abs(double val)
{
    return( fabs(val) );
}
template<>
__device__ __forceinline__ float cuda_abs(float val)
{
    return( fabsf(val) );
}


template<class T>
__device__ __forceinline__ T __GPU_REDUCTION_OGITA_H__two_asum_device(T &t, T a, T b)
{
    T s = a+T(b);
    T bs = s-a;
    T as = s-bs;
    t = (T(b)-bs) + (a-as);
    return s;
}

template<>
__device__ __forceinline__ thrust::complex<float> __GPU_REDUCTION_OGITA_H__two_asum_device(thrust::complex<float> &t, thrust::complex<float> a, thrust::complex<float> b)
{
    float t1 = float(0.0);
    thrust::complex<float> t2 = thrust::complex<float>(0.0);
    float b_real = __GPU_REDUCTION_OGITA_H__two_sum_device<float>(t1, cuda_abs<float>(b.real()), cuda_abs<float>(b.imag()) );
    thrust::complex<float> s = __GPU_REDUCTION_OGITA_H__two_sum_device< thrust::complex<float> >(t2, a, thrust::complex<float>(b_real, float(0.0)) );
    t = thrust::complex<float>(t1, float(0.0) ) + t2;
    return s;
}
template<>
__device__ __forceinline__ thrust::complex<double> __GPU_REDUCTION_OGITA_H__two_asum_device(thrust::complex<double> &t, thrust::complex<double> a, thrust::complex<double> b)
{
    double t1 = double(0.0);
    thrust::complex<double> t2 = thrust::complex<double>(0.0);
    double b_real = __GPU_REDUCTION_OGITA_H__two_sum_device<double>(t1, cuda_abs<double>(b.real()), cuda_abs<double>(b.imag()) );
    thrust::complex<double> s = __GPU_REDUCTION_OGITA_H__two_sum_device< thrust::complex<double> >(t2, a, thrust::complex<double>(b_real, double(0.0)) );
    t = thrust::complex<double>(t1, double(0.0) ) + t2;
    return s;
}

// funciton for debug =)
template<class T>
__device__ void print_var(T var)
{
    printf("var = %le\n", (double)var);
}
template<>
__device__ void print_var(thrust::complex<double> var)
{
    printf("var = (%le,%le)\n", (double)var.real(), (double)var.imag() );
}
template<>
__device__  void print_var(thrust::complex<float> var)
{
    printf("var = (%le,%le)\n", (double)var.real(), (double)var.imag() );
}
// function ends

}
#endif