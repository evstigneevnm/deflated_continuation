#ifndef __CUFFT_WRAP_H__
#define __CUFFT_WRAP_H__

#include <cufft.h>
#include <thrust/complex.h>
#include <utils/cufft_safe_call.h>

//=== type definition ===

template<typename T>
struct complex_type_hlp
{
};


template<>
struct complex_type_hlp<float>
{
    typedef cufftComplex type;
};

template<>
struct complex_type_hlp<double>
{
    typedef cufftDoubleComplex type;
};


//=== Wrap for Complex2Complex transform ===

template <typename T>
class cufft_wrap_C2C
{

};


template <>
class cufft_wrap_C2C<double>
{
public:

    typedef typename complex_type_hlp<double>::type complex_type;
    typedef typename thrust::complex<double> thrust_complex_type;

    cufft_wrap_C2C(size_t size_x): plan_created(false)
    {
        CUFFT_SAFE_CALL(cufftPlan1d(&planC2C, size_x, CUFFT_Z2Z, 1));
        plan_created=true;
    }
    cufft_wrap_C2C(size_t size_x, size_t size_y): plan_created(false)
    {
        CUFFT_SAFE_CALL(cufftPlan2d(&planC2C, size_x, size_y, CUFFT_Z2Z));
        plan_created=true;
    }
    cufft_wrap_C2C(size_t size_x, size_t size_y, size_t size_z): plan_created(false)
    {
        CUFFT_SAFE_CALL(cufftPlan3d(&planC2C, size_x, size_y, size_z, CUFFT_Z2Z));
        plan_created=true;
    }        

    ~cufft_wrap_C2C()
    {
        if(plan_created)
        {
            cufftDestroy(planC2C);
            plan_created=false;
        }
    }

    void fft(complex_type *source, complex_type *destination)
    {
        CUFFT_SAFE_CALL(cufftExecZ2Z(planC2C, source, destination, CUFFT_FORWARD));
    }

    void ifft(complex_type *source, complex_type *destination)
    {
        CUFFT_SAFE_CALL(cufftExecZ2Z(planC2C, source, destination, CUFFT_INVERSE));
    }
    
    void fft(thrust_complex_type *source, thrust_complex_type *destination)
    {
        CUFFT_SAFE_CALL(cufftExecZ2Z(planC2C, (complex_type*)source, (complex_type*)destination, CUFFT_FORWARD));
    }

    void ifft(thrust_complex_type *source, thrust_complex_type *destination)
    {
        CUFFT_SAFE_CALL(cufftExecZ2Z(planC2C, (complex_type*)source, (complex_type*)destination, CUFFT_INVERSE));
    }    

private:
    cufftHandle planC2C;
    bool plan_created;
};

template <>
class cufft_wrap_C2C<float>
{

public:
    typedef typename complex_type_hlp<float>::type complex_type;
    typedef typename thrust::complex<float> thrust_complex_type;

    cufft_wrap_C2C(size_t size_x)
    {
        CUFFT_SAFE_CALL(cufftPlan1d(&planC2C, size_x, CUFFT_C2C, 1));
    }
    cufft_wrap_C2C(size_t size_x, size_t size_y)
    {
        CUFFT_SAFE_CALL(cufftPlan2d(&planC2C, size_x, size_y, CUFFT_C2C));
    }
    cufft_wrap_C2C(size_t size_x, size_t size_y, size_t size_z)
    {
        CUFFT_SAFE_CALL(cufftPlan3d(&planC2C, size_x, size_y, size_z, CUFFT_C2C));
    }        

    ~cufft_wrap_C2C()
    {
        if(plan_created)
        {
            cufftDestroy(planC2C);
            plan_created=false;
        }
    }

    void fft(complex_type *source, complex_type *destination)
    {
        CUFFT_SAFE_CALL(cufftExecC2C(planC2C, source, destination, CUFFT_FORWARD));
    }

    void ifft(complex_type *source, complex_type *destination)
    {
        CUFFT_SAFE_CALL(cufftExecC2C(planC2C, source, destination, CUFFT_INVERSE));
    }

    void fft(thrust_complex_type *source, thrust_complex_type *destination)
    {
        CUFFT_SAFE_CALL(cufftExecC2C(planC2C, (complex_type*)source, (complex_type*)destination, CUFFT_FORWARD));
    }

    void ifft(thrust_complex_type *source, thrust_complex_type *destination)
    {
        CUFFT_SAFE_CALL(cufftExecC2C(planC2C, (complex_type*)source, (complex_type*)destination, CUFFT_INVERSE));
    }
private:
    cufftHandle planC2C;
    bool plan_created;
};




//=== Wrap for Real2Complex transform and back ===

template <typename T>
class cufft_wrap_R2C
{

};


template <>
class cufft_wrap_R2C<double>
{
public:

    typedef typename complex_type_hlp<double>::type complex_type;
    typedef typename thrust::complex<double> thrust_complex_type;

    cufft_wrap_R2C(size_t size_x): planR2C_created(false), planC2R_created(false)
    {
        CUFFT_SAFE_CALL(cufftPlan1d(&planR2C, size_x, CUFFT_D2Z, 1));
        planR2C_created=true;
        CUFFT_SAFE_CALL(cufftPlan1d(&planC2R, size_x, CUFFT_Z2D, 1));
        planC2R_created=true;
        size_j_F=floor(size_x/2)+1;
    }
    cufft_wrap_R2C(size_t size_x, size_t size_y): planR2C_created(false), planC2R_created(false)
    {
        CUFFT_SAFE_CALL(cufftPlan2d(&planR2C, size_x, size_y, CUFFT_D2Z));
        planR2C_created=true;
        CUFFT_SAFE_CALL(cufftPlan2d(&planC2R, size_x, size_y, CUFFT_Z2D));
        planC2R_created=true;
        size_j_F=floor(size_y/2)+1;
    }
    cufft_wrap_R2C(size_t size_x, size_t size_y, size_t size_z): planR2C_created(false), planC2R_created(false)
    {
        CUFFT_SAFE_CALL(cufftPlan3d(&planR2C, size_x, size_y, size_z, CUFFT_D2Z));
        planR2C_created=true;
        CUFFT_SAFE_CALL(cufftPlan3d(&planC2R, size_x, size_y, size_z, CUFFT_Z2D));
        planC2R_created=true;
        size_j_F=floor(size_z/2)+1;
    }        

    ~cufft_wrap_R2C()
    {
        if(planR2C_created)
        {
            cufftDestroy(planR2C);
            planR2C_created=false;
        }
        if(planC2R_created)
        {
            cufftDestroy(planC2R);
            planC2R_created=false;
        }

    }

    void fft(double *source, complex_type *destination)
    {
        CUFFT_SAFE_CALL(cufftExecD2Z(planR2C, source, destination));
    }

    void ifft(complex_type *source, double *destination)
    {
        CUFFT_SAFE_CALL(cufftExecZ2D(planC2R, source, destination));
    }

    void fft(double *source, thrust_complex_type *destination)
    {
        CUFFT_SAFE_CALL(cufftExecD2Z(planR2C, source, (complex_type *)destination));
    }

    void ifft(thrust_complex_type *source, double *destination)
    {
        CUFFT_SAFE_CALL(cufftExecZ2D(planC2R, (complex_type *)source, destination));
    }

    size_t get_reduced_size(){
        return size_j_F;
    }

private:
    cufftHandle planR2C;
    cufftHandle planC2R;
    size_t size_j_F;
    bool planR2C_created, planC2R_created;
};

template <>
class cufft_wrap_R2C<float>
{
public:

    typedef typename complex_type_hlp<float>::type complex_type;
    typedef typename thrust::complex<float> thrust_complex_type;

    cufft_wrap_R2C(size_t size_x): planR2C_created(false), planC2R_created(false)
    {
        CUFFT_SAFE_CALL(cufftPlan1d(&planR2C, size_x, CUFFT_R2C, 1));
        planR2C_created=true;        
        CUFFT_SAFE_CALL(cufftPlan1d(&planC2R, size_x, CUFFT_C2R, 1));
        planC2R_created=true;
        size_j_F=floor(size_x/2)+1;
    }
    cufft_wrap_R2C(size_t size_x, size_t size_y): planR2C_created(false), planC2R_created(false)
    {
        CUFFT_SAFE_CALL(cufftPlan2d(&planR2C, size_x, size_y, CUFFT_R2C));
        planR2C_created=true;        
        CUFFT_SAFE_CALL(cufftPlan2d(&planC2R, size_x, size_y, CUFFT_C2R));
        planC2R_created=true;
        size_j_F=floor(size_y/2)+1;
    }
    cufft_wrap_R2C(size_t size_x, size_t size_y, size_t size_z): planR2C_created(false), planC2R_created(false)
    {
        CUFFT_SAFE_CALL(cufftPlan3d(&planR2C, size_x, size_y, size_z, CUFFT_R2C));
        planR2C_created=true;        
        CUFFT_SAFE_CALL(cufftPlan3d(&planC2R, size_x, size_y, size_z, CUFFT_C2R));
        planC2R_created=true;
        size_j_F=floor(size_z/2)+1;
    }        

    ~cufft_wrap_R2C()
    {
        if(planR2C_created)
        {
            cufftDestroy(planR2C);
            planR2C_created=false;
        }
        if(planC2R_created)
        {
            cufftDestroy(planC2R);
            planC2R_created=false;
        }

    }

    void fft(float *source, complex_type *destination)
    {
        CUFFT_SAFE_CALL(cufftExecR2C(planR2C, source, destination));
    }

    void ifft(complex_type *source, float *destination)
    {
        CUFFT_SAFE_CALL(cufftExecC2R(planC2R, source, destination));
    }

    void fft(float *source, thrust_complex_type *destination)
    {
        CUFFT_SAFE_CALL(cufftExecR2C(planR2C, source, (complex_type*) destination));
    }

    void ifft(thrust_complex_type *source, float *destination)
    {
        CUFFT_SAFE_CALL(cufftExecC2R(planC2R, (complex_type*) source, destination));
    }

    size_t get_reduced_size(){
        return size_j_F;
    }

    

private:
    cufftHandle planR2C;
    cufftHandle planC2R;
    size_t size_j_F;
    bool planR2C_created, planC2R_created;

};







#endif