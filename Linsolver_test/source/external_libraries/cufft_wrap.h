#ifndef __CUFFT_WRAP_H__
#define __CUFFT_WRAP_H__

#include <cufft.h>
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


//=== Wrap for Complex 2 Complex transform ===

template <typename T>
class cufft_wrap_C2C
{

};


template <>
class cufft_wrap_C2C<double>
{
public:

    typedef typename complex_type_hlp<double>::type complex_type;

    cufft_wrap_C2C(size_t size_x)
    {
        CUFFT_SAFE_CALL(cufftPlan1d(&planC2C, size_x, CUFFT_Z2Z, 1));
    }
    cufft_wrap_C2C(size_t size_x, size_t size_y)
    {
        CUFFT_SAFE_CALL(cufftPlan2d(&planC2C, size_x, size_y, CUFFT_Z2Z));
    }
    cufft_wrap_C2C(size_t size_x, size_t size_y, size_t size_z)
    {
        CUFFT_SAFE_CALL(cufftPlan3d(&planC2C, size_x, size_y, size_z, CUFFT_Z2Z));
    }        

    ~cufft_wrap_C2C(){
        cufftDestroy(planC2C);
    }

    void fft(complex_type *source, complex_type *destination)
    {
        CUFFT_SAFE_CALL(cufftExecZ2Z(planC2C, source, destination, CUFFT_FORWARD));
    }

    void ifft(complex_type *source, complex_type *destination)
    {
        CUFFT_SAFE_CALL(cufftExecZ2Z(planC2C, source, destination, CUFFT_INVERSE));
    }

protected:
    cufftHandle planC2C;

};

template <>
class cufft_wrap_C2C<float>
{

public:
    typedef typename complex_type_hlp<float>::type complex_type;

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

    ~cufft_wrap_C2C(){
        cufftDestroy(planC2C);
    }

    void fft(complex_type *source, complex_type *destination)
    {
        CUFFT_SAFE_CALL(cufftExecC2C(planC2C, source, destination, CUFFT_FORWARD));
    }

    void ifft(complex_type *source, complex_type *destination)
    {
        CUFFT_SAFE_CALL(cufftExecC2C(planC2C, source, destination, CUFFT_INVERSE));
    }

protected:
    cufftHandle planC2C;

};




//=== Wrap for Real 2 Complex transform and back ===

template <typename T>
class cufft_wrap_R2C
{

};


template <>
class cufft_wrap_R2C<double>
{
public:

    typedef typename complex_type_hlp<double>::type complex_type;

    cufft_wrap_R2C(size_t size_x)
    {
        CUFFT_SAFE_CALL(cufftPlan1d(&planR2C, size_x, CUFFT_D2Z, 1));
        CUFFT_SAFE_CALL(cufftPlan1d(&planC2R, size_x, CUFFT_Z2D, 1));
        size_j_F=floor(size_x/2)+1;
    }
    cufft_wrap_R2C(size_t size_x, size_t size_y)
    {
        CUFFT_SAFE_CALL(cufftPlan2d(&planR2C, size_x, size_y, CUFFT_D2Z));
        CUFFT_SAFE_CALL(cufftPlan2d(&planC2R, size_x, size_y, CUFFT_Z2D));
        size_j_F=floor(size_y/2)+1;
    }
    cufft_wrap_R2C(size_t size_x, size_t size_y, size_t size_z)
    {
        CUFFT_SAFE_CALL(cufftPlan3d(&planR2C, size_x, size_y, size_z, CUFFT_D2Z));
        CUFFT_SAFE_CALL(cufftPlan3d(&planC2R, size_x, size_y, size_z, CUFFT_Z2D));
        size_j_F=floor(size_z/2)+1;
    }        

    ~cufft_wrap_R2C(){
        cufftDestroy(planR2C);
        cufftDestroy(planC2R);
    }

    void fft(double *source, complex_type *destination)
    {
        CUFFT_SAFE_CALL(cufftExecD2Z(planR2C, source, destination));
    }

    void ifft(complex_type *source, double *destination)
    {
        CUFFT_SAFE_CALL(cufftExecZ2D(planC2R, source, destination));
    }

    size_t get_reduced_size(){
        return size_j_F;
    }

protected:
    cufftHandle planR2C;
    cufftHandle planC2R;
private:
    size_t size_j_F;
};

template <>
class cufft_wrap_R2C<float>
{
public:

    typedef typename complex_type_hlp<float>::type complex_type;

    cufft_wrap_R2C(size_t size_x)
    {
        CUFFT_SAFE_CALL(cufftPlan1d(&planR2C, size_x, CUFFT_R2C, 1));
        CUFFT_SAFE_CALL(cufftPlan1d(&planC2R, size_x, CUFFT_C2R, 1));
        size_j_F=floor(size_x/2)+1;
    }
    cufft_wrap_R2C(size_t size_x, size_t size_y)
    {
        CUFFT_SAFE_CALL(cufftPlan2d(&planR2C, size_x, size_y, CUFFT_R2C));
        CUFFT_SAFE_CALL(cufftPlan2d(&planC2R, size_x, size_y, CUFFT_C2R));
        size_j_F=floor(size_y/2)+1;
    }
    cufft_wrap_R2C(size_t size_x, size_t size_y, size_t size_z)
    {
        CUFFT_SAFE_CALL(cufftPlan3d(&planR2C, size_x, size_y, size_z, CUFFT_R2C));
        CUFFT_SAFE_CALL(cufftPlan3d(&planC2R, size_x, size_y, size_z, CUFFT_C2R));
        size_j_F=floor(size_z/2)+1;
    }        

    ~cufft_wrap_R2C(){
        cufftDestroy(planR2C);
        cufftDestroy(planC2R);
    }

    void fft(float *source, complex_type *destination)
    {
        CUFFT_SAFE_CALL(cufftExecR2C(planR2C, source, destination));
    }

    void ifft(complex_type *source, float *destination)
    {
        CUFFT_SAFE_CALL(cufftExecC2R(planC2R, source, destination));
    }

    size_t get_reduced_size(){
        return size_j_F;
    }

protected:
    cufftHandle planR2C;
    cufftHandle planC2R;
private:
    size_t size_j_F;
};


#endif