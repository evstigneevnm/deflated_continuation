/*

https://docs.nvidia.com/cuda/cublas/index.html

For maximum compatibility with existing Fortran environments, the cuBLAS library uses column-major storage, 
and 1-based indexing. 
Since C and C++ use row-major storage, applications written in these languages can not use the native array 
semantics for two-dimensional arrays. Instead, macros or inline functions should be defined to implement 
matrices on top of one-dimensional arrays. For Fortran code ported to C in mechanical fashion, one may chose 
to retain 1-based indexing to avoid the need to transform loops. In this case, the array index of a matrix 
element in row ''i'' and column ''j'' can be computed via the following macro:

#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))

Here, ld refers to the leading dimension of the matrix, which in the case of column-major storage is the 
number of rows of the allocated matrix (even if only a submatrix of it is being used). 
For natively written C and C++ code, one would most likely choose 0-based indexing, in which case the array 
index of a matrix element in row ''i'' and column 'j'' can be computed via the following macro:

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

*/


#ifndef __CUBLAS_WRAP_H__
#define __CUBLAS_WRAP_H__

#include <iostream>
#include <cublas_v2.h>
#include <thrust/complex.h>
#include <utils/cublas_safe_call.h>
#include <stdexcept>

namespace cublas_complex_types{
    template<typename T>
    struct cublas_cuComplex_type_hlp
    {
    };


    template<>
    struct cublas_cuComplex_type_hlp<float>
    {
        typedef cuComplex type;
    };

    template<>
    struct cublas_cuComplex_type_hlp<double>
    {
        typedef cuDoubleComplex type;
    };
}

namespace cublas_real_types{
    template<typename T>
    struct cublas_real_type_hlp
    {
    };

    template<>
    struct cublas_real_type_hlp<float>
    {
        typedef float type;
    };

    template<>
    struct cublas_real_type_hlp<double>
    {
        typedef double type;
    };

    template<>
    struct cublas_real_type_hlp<cuComplex>
    {
        typedef float type;
    };

    template<>
    struct cublas_real_type_hlp<cuDoubleComplex>
    {
        typedef double type;
    };

    template<>
    struct cublas_real_type_hlp< thrust::complex<float> >
    {
        typedef float type;
    };

    template<>
    struct cublas_real_type_hlp< thrust::complex<double> >
    {
        typedef double type;
    };    
}

class cublas_wrap
{
public:


    cublas_wrap(): handle_created(false)
    {
        cublas_create();
        handle_created=true;
        set_pointer_location_device(false);
    }

    cublas_wrap(bool plot_info): handle_created(false)
    {
        if(plot_info)
        {
            cublas_create_info();
            handle_created=true;
        }
        else
        {
            cublas_create();
            handle_created=true;
        }
        set_pointer_location_device(false);
    }


    ~cublas_wrap()
    {
        if(handle_created)
        {
            cublas_destroy();
            handle_created=false;
        }
    }
 
    cublasHandle_t* get_handle()
    {
        return &handle;
    }


    void set_stream(cudaStream_t streamId)
    {
        CUBLAS_SAFE_CALL(cublasSetStream(handle, streamId));
    }

    cudaStream_t get_stream()
    {
        cudaStream_t streamId;
        CUBLAS_SAFE_CALL(cublasGetStream(handle, &streamId));
        return streamId;
    }
    
    //where scalar results are stored, like dot product etc.
    cublasPointerMode_t get_pointer_location()
    {
        cublasPointerMode_t mode;
        CUBLAS_SAFE_CALL(cublasGetPointerMode(handle, &mode));
        return mode;
    }

    //where to store scalar results, like dot product etc.
    // true means store on device
    //this scalar pointer must be allocated on the device via cudaMalloc!
    void set_pointer_location_device(bool store_on_device)
    {
        if(store_on_device){
            CUBLAS_SAFE_CALL(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE)); 
            scalar_pointer_on_device=true;
        }
        else{
            CUBLAS_SAFE_CALL(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST)); 
            scalar_pointer_on_device=false;
        }
    }

    template<typename T>
    void set_vector(size_t vector_size, const T *vec_host, T *vec_device, int incx=1, int incy=1)
    {
        CUBLAS_SAFE_CALL(cublasSetVector(vector_size, sizeof(T), vec_host, incx, vec_device, incy));
    }
    template<typename T>
    void get_vector(size_t vector_size, const T *vec_device,  T *vec_host, int incx=1, int incy=1)
    {
        CUBLAS_SAFE_CALL(cublasGetVector(vector_size, sizeof(T), vec_device, incx, vec_host, incy));
    }   
    template<typename T>
    void set_matrix(size_t rows, size_t cols, const T *mat_host, int lda, T *mat_device, int ldb)
    {
        CUBLAS_SAFE_CALL(cublasSetMatrix(rows, cols, sizeof(T), mat_host, lda, mat_device, ldb));
    }
    template<typename T>
    void get_matrix(size_t rows, size_t cols, const T *mat_device, int lda, T *mat_host,  int ldb)
    {
        CUBLAS_SAFE_CALL(cublasGetMatrix(rows, cols, sizeof(T), mat_device, lda, mat_host, ldb));
    }
    template<typename T>
    void set_vector_async(size_t vector_size, const T *vec_host, T *vec_device, cudaStream_t stream, int incx=1, int incy=1)
    {
        CUBLAS_SAFE_CALL(cublasSetVectorAsync(vector_size, sizeof(T), vec_host, incx, vec_device, incy, stream));
    }
    template<typename T>
    void get_vector_async(size_t vector_size, const T *vec_device,  T *vec_host, cudaStream_t stream, int incx=1, int incy=1)
    {
        CUBLAS_SAFE_CALL(cublasGetVectorAsync(vector_size, sizeof(T), vec_device, incx, vec_host, incy, stream));
    }   
    template<typename T>
    void set_matrix_async(size_t rows, size_t cols, const T *mat_host, T *mat_device, cudaStream_t stream,  int lda=1, int ldb=1)
    {
        CUBLAS_SAFE_CALL(cublasSetMatrixAsync(rows, cols, sizeof(T), mat_host, lda, mat_device, ldb, stream));
    }
    template<typename T>
    void get_matrix_async(size_t rows, size_t cols, const T *mat_device,  T *mat_host, cudaStream_t stream,  int lda=1, int ldb=1)
    {
        CUBLAS_SAFE_CALL(cublasGetMatrixAsync(rows, cols, sizeof(T), mat_device, lda, mat_host, ldb, stream));
    }
    
    //TODO Can't be used in cuda 8?!? WTF? see: https://docs.nvidia.com/cuda/cublas/index.html#cublasmath_t
    // void use_tensor_core_operations(bool useTCO)
    // {
    //     cublasMath_t mode = CUBLAS_DEFAULT_MATH;
    //     if(useTCO)
    //         mode = CUBLAS_TENSOR_OP_MATH;

    //     CUBLAS_SAFE_CALL(cublasSetMathMode(handle, mode));
        
    // }

    //===cuBLAS Level-1 Functions=== see: https://docs.nvidia.com/cuda/cublas/index.html#cublas-level-1-function-reference
    template<typename T>
    void sum_abs_elements(size_t vector_size, const T *vector, 
                            typename cublas_real_types::cublas_real_type_hlp<T>::type *result, int incx=1);

    // y [ j ] = alpha x [ k ] + y [ j ]
    template<typename T>
    void axpy(size_t vector_sizes, const T alpha, const T *x, T *y, int incx=1, int incy=1);
    // y [ j ] = x [ k ] 
    template<typename T>
    void copy(size_t vector_sizes, const T *x, T *y, int incx=1, int incy=1);
    // y [ j ] <-> x [ k ] 
    template<typename T>
    void swap(size_t vector_sizes, T *x, T *y, int incx=1, int incy=1);
    //dot product (we use automatic conjugation for complex number, i.e. dot(u,v)=u^C*v)
    template<typename T>
    void dot(size_t vector_size, const T *x, const T *y, T *result, int incx=1, int incy=1);
    //vector l2 norm.
    template<typename T>
    void norm2(size_t vector_size, const T *x, typename cublas_real_types::cublas_real_type_hlp<T>::type *result, int incx=1);
    //scale vector as x=x*a. 'a' can be real or complex
    template<typename T>
    void scale(size_t vector_size, const T alpha, T *x, int incx=1);
    template<typename T>
    void scale(size_t vector_size, const typename cublas_real_types::cublas_real_type_hlp<T>::type alpha, T *x, int incx=1);
    //normalizes vector. Overwrites a vector with normalized one and returns it's norm. Usually used in Krylov-type methods
    template<typename T>
    void normalize(size_t vector_size, T *x, T *norm, int incx=1);
    template<typename T>
    void normalize(size_t vector_size, T *x, typename cublas_real_types::cublas_real_type_hlp<T>::type *norm, int incx=1);
    //TODO: add Givens rotations construction for Arnoldi process

    //===cuBLAS Level-2 Functions=== see: https://docs.nvidia.com/cuda/cublas/index.html#cublas-level-2-function-reference
private:
    cublasOperation_t switch_operation_real(const char& op)
    {
        cublasOperation_t operation = CUBLAS_OP_N;
        switch(op)
        {
            case 'N':
                operation = CUBLAS_OP_N;
                break;
            case 'T':
                operation = CUBLAS_OP_T;
                break;
            default:
                // invalid operation code throw
                throw std::runtime_error("switch_operation_real: invalid code for original or transpose operations. Only 'N' or 'T' are defined.");
                break;                   
        }  
        return operation;

    }

    cublasOperation_t switch_operation_complex(const char& op)
    {
        cublasOperation_t operation = CUBLAS_OP_N;
        switch(op)
        {
            case 'N':
                operation = CUBLAS_OP_N;
                break;
            case 'T':
                operation = CUBLAS_OP_C;// CUBLAS_OP_H is defined in documentaiton?!? 
                                        //definition in:
                                        // ../cuda/include/cublas_api.h
                break;    
            default:
                // invalid operation code throw
                throw std::runtime_error("switch_operation_complex: invalid code for original or transpose operations. Only 'N' or 'T' (for Hermitian transpose) are defined.");                
                break;                              
        }  
        return operation;

    }
public:    
    //This function performs the matrix-vector multiplication: 
    //                  y = α op ( A ) x + β y,
    //where A is a m × n matrix stored in column-major format, x and y are vectors, and α and β are scalars. 
    //Also, for matrix A:
    //   op(A) = A  if transa == CUBLAS_OP_N 
    //   op(A) = A^T  if transa == CUBLAS_OP_T 
    //   op(A) = A^H  if transa == CUBLAS_OP_H 
    //   op = 'N' for CUBLAS_OP_N, op = 'T' for CUBLAS_OP_T, op = 'H' for CUBLAS_OP_H
    //LDimA is the leading dimension of A, for C arrays LDimA = RowA.
    template<typename T>
    void gemv(const char op, size_t RowA, const T *A, size_t ColA, size_t LDimA, const T alpha, const T *x, const T beta, T *y);


    //===cuBLAS Level-3 Functions=== see: https://docs.nvidia.com/cuda/cublas/index.html#cublas-level-3-function-reference
    template<typename T>
    void gemm(const char opA, const char opB, size_t RowAC, size_t ColBC, size_t ColARowB, const T alpha, const T* A, size_t LDimA, const T* B, size_t LDimB, const T beta, T* C, size_t LdimC);

    //===cuBLAS BLAS-like EXTENSIONS=== see: https://docs.nvidia.com/cuda/cublas/index.html#blas-like-extension
    //TODO: to be inplemented on demand.

private:
    cublasHandle_t handle;
    bool handle_created;
    bool scalar_pointer_on_device;



    void cublas_create()
    {
        
        CUBLAS_SAFE_CALL(cublasCreate(&handle));

    }

    void cublas_destroy()
    {
        
        CUBLAS_SAFE_CALL(cublasDestroy(handle));

    }

    void cublas_create_info()
    {
        
        int cublas_version, major_ver, minor_ver, patch_level;
        CUBLAS_SAFE_CALL(cublasCreate(&handle));
        CUBLAS_SAFE_CALL(cublasGetVersion(handle, &cublas_version));
        CUBLAS_SAFE_CALL(cublasGetProperty(MAJOR_VERSION, &major_ver));
        CUBLAS_SAFE_CALL(cublasGetProperty(MINOR_VERSION, &minor_ver));
        CUBLAS_SAFE_CALL(cublasGetProperty(PATCH_LEVEL, &patch_level));

        std::cout << "cuBLAS v."<< cublas_version << " (major="<< major_ver << ", minor=" << minor_ver << ", patch level=" << patch_level << ") handle created." << std::endl;

    }



};

// template specializations for level 1 BLAS functions

template<> inline
void cublas_wrap::sum_abs_elements(size_t vector_size, const float *vector, float *result, int incx)
{
    CUBLAS_SAFE_CALL(cublasSasum(handle, vector_size, vector, incx, result));
}
template<> inline
void cublas_wrap::sum_abs_elements(size_t vector_size, const double *vector, double *result, int incx)
{
    CUBLAS_SAFE_CALL(cublasDasum(handle, vector_size, vector, incx, result));
}
template<> inline
void cublas_wrap::sum_abs_elements(size_t vector_size, const thrust::complex<float> *vector, typename cublas_real_types::cublas_real_type_hlp< thrust::complex<float> >::type *result, int incx)
{
    CUBLAS_SAFE_CALL(cublasScasum(handle, vector_size, (cuComplex*) vector, incx, result));
}    
template<> inline
void cublas_wrap::sum_abs_elements(size_t vector_size, const thrust::complex<double> *vector, typename cublas_real_types::cublas_real_type_hlp< thrust::complex<double> >::type *result, int incx)
{

    CUBLAS_SAFE_CALL(cublasDzasum(handle, vector_size, (cuDoubleComplex*) vector, incx, result));

}   
template<> inline
void cublas_wrap::sum_abs_elements(size_t vector_size, const cuComplex *vector, typename cublas_real_types::cublas_real_type_hlp< cuComplex >::type *result, int incx)
{
    CUBLAS_SAFE_CALL(cublasScasum(handle, vector_size, vector, incx, result));
}    
template<> inline
void cublas_wrap::sum_abs_elements(size_t vector_size, const cuDoubleComplex *vector, typename cublas_real_types::cublas_real_type_hlp< cuDoubleComplex >::type *result, int incx)
{

    CUBLAS_SAFE_CALL(cublasDzasum(handle, vector_size, vector, incx, result));

}       
//This function multiplies the vector x by the scalar alpha 
//and adds it to the vector y overwriting the latest vector with the result.
//y=alpha*x+y;
template<> inline
void cublas_wrap::axpy(size_t vector_sizes, const float alpha, const float *x, float *y, int incx, int incy)
{
    CUBLAS_SAFE_CALL(cublasSaxpy(handle, vector_sizes, &alpha, x, incx, y, incy));
}
template<> inline
void cublas_wrap::axpy(size_t vector_sizes, const double alpha, const double *x, double *y, int incx, int incy)
{
    CUBLAS_SAFE_CALL(cublasDaxpy(handle, vector_sizes, &alpha, x, incx, y, incy));
}
template<> inline
void cublas_wrap::axpy(size_t vector_sizes, const cuComplex alpha, const cuComplex *x, cuComplex *y, int incx, int incy)
{
    CUBLAS_SAFE_CALL(cublasCaxpy(handle, vector_sizes, &alpha, x, incx, y, incy));
}
template<> inline
void cublas_wrap::axpy(size_t vector_sizes, const cuDoubleComplex alpha, const cuDoubleComplex *x, cuDoubleComplex *y, int incx, int incy)
{
    CUBLAS_SAFE_CALL(cublasZaxpy(handle, vector_sizes, &alpha, x, incx, y, incy));
}
template<> inline
void cublas_wrap::axpy(size_t vector_sizes, const thrust::complex<float> alpha, const thrust::complex<float> *x, thrust::complex<float> *y, int incx, int incy)
{
    CUBLAS_SAFE_CALL(cublasCaxpy(handle, vector_sizes, (cuComplex*)&alpha, (cuComplex*)x, incx, (cuComplex*)y, incy));
}
template<> inline
void cublas_wrap::axpy(size_t vector_sizes, const thrust::complex<double> alpha, const thrust::complex<double> *x, thrust::complex<double> *y, int incx, int incy)
{
    CUBLAS_SAFE_CALL(cublasZaxpy(handle, vector_sizes, (cuDoubleComplex*)&alpha, (cuDoubleComplex*)x, incx, (cuDoubleComplex*)y, incy));
}
//
template<> inline
void cublas_wrap::copy(size_t vector_sizes, const float *x, float *y, int incx, int incy)
{
    CUBLAS_SAFE_CALL(cublasScopy(handle, vector_sizes, x, incx, y, incy));
}
template<> inline
void cublas_wrap::copy(size_t vector_sizes, const double *x, double *y, int incx, int incy)
{
    CUBLAS_SAFE_CALL(cublasDcopy(handle, vector_sizes, x, incx, y, incy));
}
template<> inline
void cublas_wrap::copy(size_t vector_sizes, const cuComplex *x, cuComplex *y, int incx, int incy)
{
    CUBLAS_SAFE_CALL(cublasCcopy(handle, vector_sizes, x, incx, y, incy));
}
template<> inline
void cublas_wrap::copy(size_t vector_sizes, const cuDoubleComplex *x, cuDoubleComplex *y, int incx, int incy)
{
    CUBLAS_SAFE_CALL(cublasZcopy(handle, vector_sizes, x, incx, y, incy));
}
template<> inline
void cublas_wrap::copy(size_t vector_sizes, const thrust::complex<float> *x, thrust::complex<float> *y, int incx, int incy)
{
    CUBLAS_SAFE_CALL(cublasCcopy(handle, vector_sizes, (cuComplex*)x, incx, (cuComplex*)y, incy));
}
template<> inline
void cublas_wrap::copy(size_t vector_sizes, const thrust::complex<double> *x, thrust::complex<double> *y, int incx, int incy)
{
    CUBLAS_SAFE_CALL(cublasZcopy(handle, vector_sizes, (cuDoubleComplex*)x, incx, (cuDoubleComplex*)y, incy));
}
//
template<> inline
void cublas_wrap::swap(size_t vector_size, float *x, float *y, int incx, int incy)
{
    CUBLAS_SAFE_CALL(cublasSswap(handle, vector_size, x, incx, y, incy));
}
template<> inline
void cublas_wrap::swap(size_t vector_size, double *x, double *y, int incx, int incy)
{
    CUBLAS_SAFE_CALL(cublasDswap(handle, vector_size, x, incx, y, incy));
}
template<> inline
void cublas_wrap::swap(size_t vector_size, cuComplex *x, cuComplex *y, int incx, int incy)
{
    CUBLAS_SAFE_CALL(cublasCswap(handle, vector_size, x, incx, y, incy));
}    
template<> inline
void cublas_wrap::swap(size_t vector_size, cuDoubleComplex *x, cuDoubleComplex *y, int incx, int incy)
{
    CUBLAS_SAFE_CALL(cublasZswap(handle, vector_size, x, incx, y, incy));
}   
template<> inline
void cublas_wrap::swap(size_t vector_size, thrust::complex<float> *x, thrust::complex<float> *y, int incx, int incy)
{
    CUBLAS_SAFE_CALL(cublasCswap(handle, vector_size, (cuComplex*)x, incx, (cuComplex*)y, incy));
}    
template<> inline
void cublas_wrap::swap(size_t vector_size, thrust::complex<double> *x, thrust::complex<double> *y, int incx, int incy)
{
    CUBLAS_SAFE_CALL(cublasZswap(handle, vector_size, (cuDoubleComplex*)x, incx, (cuDoubleComplex*)y, incy));
}       
//
template<> inline
void cublas_wrap::dot(size_t vector_size, const float *x, const float *y, float *result, int incx, int incy)
{
    CUBLAS_SAFE_CALL(cublasSdot (handle, vector_size, x, incx, y, incy, result));
}
template<> inline
void cublas_wrap::dot(size_t vector_size, const double *x, const double *y, double *result, int incx, int incy)
{
    CUBLAS_SAFE_CALL(cublasDdot (handle, vector_size, x, incx, y, incy, result));
}    
template<> inline
void cublas_wrap::dot(size_t vector_size, const cuComplex *x, const cuComplex *y, cuComplex *result, int incx, int incy)
{
    CUBLAS_SAFE_CALL(cublasCdotc (handle, vector_size, x, incx, y, incy, result));
}
template<> inline
void cublas_wrap::dot(size_t vector_size, const cuDoubleComplex *x, const cuDoubleComplex *y, cuDoubleComplex *result, int incx, int incy)
{
    CUBLAS_SAFE_CALL(cublasZdotc (handle, vector_size, x, incx, y, incy, result));
}    
template<> inline
void cublas_wrap::dot(size_t vector_size, const thrust::complex<float> *x, const thrust::complex<float> *y,  thrust::complex<float> *result, int incx, int incy)
{
    CUBLAS_SAFE_CALL(cublasCdotc (handle, vector_size, (cuComplex*)x, incx, (cuComplex*)y, incy, (cuComplex*)result));
}
template<> inline
void cublas_wrap::dot(size_t vector_size, const thrust::complex<double> *x, const thrust::complex<double> *y, thrust::complex<double> *result, int incx, int incy)
{
    CUBLAS_SAFE_CALL(cublasZdotc (handle, vector_size, (cuDoubleComplex*)x, incx, (cuDoubleComplex*)y, incy, (cuDoubleComplex*)result));
}  
//
template<> inline
void cublas_wrap::norm2(size_t vector_size, const float *x, float *result, int incx)
{
    CUBLAS_SAFE_CALL(cublasSnrm2(handle, vector_size, x, incx, result));
}
template<> inline
void cublas_wrap::norm2(size_t vector_size, const double *x, double *result, int incx)
{
    CUBLAS_SAFE_CALL(cublasDnrm2(handle, vector_size, x, incx, result));
}
template<> inline
void cublas_wrap::norm2(size_t vector_size, const cuComplex *x, typename cublas_real_types::cublas_real_type_hlp< cuComplex >::type *result, int incx)
{
    CUBLAS_SAFE_CALL(cublasScnrm2(handle, vector_size, x, incx, result));
}
template<> inline
void cublas_wrap::norm2(size_t vector_size, const cuDoubleComplex *x, typename cublas_real_types::cublas_real_type_hlp< cuDoubleComplex >::type *result, int incx)
{
    CUBLAS_SAFE_CALL(cublasDznrm2(handle, vector_size, x, incx, result));
}
template<> inline
void cublas_wrap::norm2(size_t vector_size, const thrust::complex<float>  *x, typename cublas_real_types::cublas_real_type_hlp< thrust::complex<float> >::type *result, int incx)
{
    CUBLAS_SAFE_CALL(cublasScnrm2(handle, vector_size, (cuComplex*)x, incx, result));
}
template<> inline
void cublas_wrap::norm2(size_t vector_size, const thrust::complex<double> *x, typename cublas_real_types::cublas_real_type_hlp< thrust::complex<double> >::type *result, int incx)
{
    CUBLAS_SAFE_CALL(cublasDznrm2(handle, vector_size, (cuDoubleComplex*)x, incx, result));
}
//
template<> inline
void cublas_wrap::scale(size_t vector_size, const float alpha, float *x, int incx)
{
    CUBLAS_SAFE_CALL(cublasSscal(handle, vector_size, &alpha, x, incx));
}
template<> inline
void cublas_wrap::scale(size_t vector_size, const double alpha, double *x, int incx)
{
    CUBLAS_SAFE_CALL(cublasDscal(handle, vector_size, &alpha, x, incx));
}
template<> inline
void cublas_wrap::scale(size_t vector_size, const cuComplex alpha, cuComplex *x, int incx)
{
    CUBLAS_SAFE_CALL(cublasCscal(handle, vector_size, &alpha, x, incx));
}
template<> inline
void cublas_wrap::scale(size_t vector_size, const cuDoubleComplex alpha, cuDoubleComplex *x, int incx)
{
    CUBLAS_SAFE_CALL(cublasZscal(handle, vector_size, &alpha, x, incx));
}
template<> inline
void cublas_wrap::scale(size_t vector_size, const thrust::complex<float> alpha, thrust::complex<float> *x, int incx)
{
    CUBLAS_SAFE_CALL(cublasCscal(handle, vector_size, (cuComplex*)&alpha, (cuComplex*)x, incx));
}
template<> inline
void cublas_wrap::scale(size_t vector_size, const thrust::complex<double> alpha, thrust::complex<double> *x, int incx)
{
    CUBLAS_SAFE_CALL(cublasZscal(handle, vector_size, (cuDoubleComplex*)&alpha, (cuDoubleComplex*)x, incx));
}
template<> inline
void cublas_wrap::scale(size_t vector_size, const float alpha, cuComplex *x, int incx)
{
    CUBLAS_SAFE_CALL(cublasCsscal(handle, vector_size, &alpha, x, incx));
}
template<> inline
void cublas_wrap::scale(size_t vector_size, const double alpha, cuDoubleComplex *x, int incx)
{
    CUBLAS_SAFE_CALL(cublasZdscal(handle, vector_size, &alpha, x, incx));
}
template<> inline
void cublas_wrap::scale(size_t vector_size, const float alpha, thrust::complex<float> *x, int incx)
{
    CUBLAS_SAFE_CALL(cublasCsscal(handle, vector_size, &alpha, (cuComplex*)x, incx));
}
template<> inline
void cublas_wrap::scale(size_t vector_size, const double alpha, thrust::complex<double> *x, int incx)
{
    CUBLAS_SAFE_CALL(cublasZdscal(handle, vector_size, &alpha, (cuDoubleComplex*)x, incx));
}
// aditional functions that are common
template<> inline
void cublas_wrap::normalize(size_t vector_size, float *x, float *norm, int incx)
{
    norm2<float>(vector_size, (const float*)x, norm, incx);
    if(scalar_pointer_on_device)
    {

    }
    else
    {
        float inorm=float(1.0)/norm[0];
        scale<float>(vector_size, (const float) inorm, x, incx);    
    }
    
}
template<> inline
void cublas_wrap::normalize(size_t vector_size, double *x, double *norm, int incx)
{
    norm2<double>(vector_size, (const double*)x, norm, incx);
    if(scalar_pointer_on_device)
    {

    }
    else
    {
        double inorm=float(1.0)/norm[0];
        scale<double>(vector_size, (const double) inorm, x, incx);    
    }        
}
template<> inline
void cublas_wrap::normalize(size_t vector_size, cuComplex *x, typename cublas_real_types::cublas_real_type_hlp< cuComplex >::type *norm, int incx)
{
    norm2<cuComplex>(vector_size, (const cuComplex*)x, norm, incx);
    if(scalar_pointer_on_device)
    {

    }
    else
    {
        float inorm=float(1.0)/norm[0];
        scale<cuComplex>(vector_size, (const float) inorm, x, incx);    
    }
}
template<> inline
void cublas_wrap::normalize(size_t vector_size, cuDoubleComplex *x, typename cublas_real_types::cublas_real_type_hlp< cuDoubleComplex >::type *norm, int incx)
{
    norm2<cuDoubleComplex>(vector_size, (const cuDoubleComplex*)x, norm, incx);
    if(scalar_pointer_on_device)
    {

    }
    else
    {
        double inorm=double(1.0)/norm[0];
        scale<cuDoubleComplex>(vector_size, (const double) inorm, x, incx);    
    }        
}
template<> inline
void cublas_wrap::normalize(size_t vector_size, thrust::complex<float> *x, typename cublas_real_types::cublas_real_type_hlp< thrust::complex<float> >::type *norm, int incx)
{
    norm2< thrust::complex<float> >(vector_size, x, norm, incx);
    if(scalar_pointer_on_device)
    {

    }
    else
    {
        float inorm=float(1.0)/norm[0];
        scale< thrust::complex<float> >(vector_size, (const float) inorm, x, incx);
    }         

}
template<> inline
void cublas_wrap::normalize(size_t vector_size, thrust::complex<double> *x, typename cublas_real_types::cublas_real_type_hlp< thrust::complex<double> >::type *norm, int incx)
{
    norm2< thrust::complex<double> >(vector_size, x, norm, incx);
    if(scalar_pointer_on_device)
    {

    }
    else
    {
        double inorm=double(1.0)/norm[0];
        scale< thrust::complex<double> >(vector_size, (const double) inorm, x, incx);
    }
}


//level 2 BLAS specializations:

template<> inline
void cublas_wrap::gemv(const char op, size_t RowA, const float *A, size_t ColA, size_t LDimA, const float alpha, const float *x, const float beta, float *y)
{

/*
cublasStatus_t cublasSgemv(cublasHandle_t handle, 
                           cublasOperation_t trans,
                           int m, 
                           int n,
                           const float *alpha,
                           const float *A, 
                           int lda,
                           const float *x, 
                           int incx,
                           const float *beta,
                           float *y, 
                           int incy)
*/    
    CUBLAS_SAFE_CALL( cublasSgemv(handle, switch_operation_real(op), RowA, ColA, &alpha, A, LDimA, x, 1, &beta, y, 1) );

}

template<> inline
void cublas_wrap::gemv(const char op, size_t RowA, const double *A, size_t ColA, size_t LDimA, const double alpha, const double *x, const double beta, double *y)
{

    CUBLAS_SAFE_CALL( cublasDgemv(handle, switch_operation_real(op), RowA, ColA, &alpha, A, LDimA, x, 1, &beta, y, 1) );

}
template<> inline
void cublas_wrap::gemv(const char op, size_t RowA, const thrust::complex<float> *A, size_t ColA, size_t LDimA, const thrust::complex<float> alpha, const thrust::complex<float> *x, const thrust::complex<float> beta, thrust::complex<float> *y)
{

    CUBLAS_SAFE_CALL( cublasCgemv(handle, switch_operation_complex(op), RowA, ColA, (const cuComplex*) &alpha, (const cuComplex*) A, LDimA, (const cuComplex*)x, 1, (const cuComplex*) &beta, (cuComplex*)y, 1) );

}
template<> inline
void cublas_wrap::gemv(const char op, size_t RowA, const thrust::complex<double> *A, size_t ColA, size_t LDimA, const thrust::complex<double> alpha, const thrust::complex<double> *x, const thrust::complex<double> beta, thrust::complex<double> *y)
{

    CUBLAS_SAFE_CALL( cublasZgemv(handle, switch_operation_complex(op), RowA, ColA, (const cuDoubleComplex*) &alpha, (const cuDoubleComplex*) A, LDimA, (const cuDoubleComplex*) x, 1, (const cuDoubleComplex*) &beta, (cuDoubleComplex*) y, 1) );

}

//level 3 BLAS specializations:
template<> inline
void cublas_wrap::gemm(const char opA, const char opB, size_t RowA, size_t ColBC, size_t ColARowB, const float alpha, const float* A, size_t LDimA, const float* B, size_t LDimB, const float beta, float* C, size_t LDimC)
{
/*
 C = α op ( A ) op ( B ) + β C 

cublasStatus_t cublasDgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const double          *alpha,
                           const double          *A, int lda,
                           const double          *B, int ldb,
                           const double          *beta,
                           double          *C, int ldc)

    m - number of rows of matrix op(A) and C.

    n - number of columns of matrix op(B) and C.
    
    k - number of columns of op(A) and rows of op(B). 

    lda - leading dimension of two-dimensional array used to store the matrix A. 

    ldb - leading dimension of two-dimensional array used to store matrix B. 

    ldc - leading dimension of a two-dimensional array used to store the matrix C. 

*/

   CUBLAS_SAFE_CALL( cublasSgemm(handle, switch_operation_real(opA), switch_operation_real(opB),
                           RowA, ColBC, ColARowB,
                           &alpha,
                           A, LDimA,
                           B, LDimB,
                           &beta,
                           C, LDimC) ); 


}

template<> inline
void cublas_wrap::gemm(const char opA, const char opB, size_t RowA, size_t ColBC, size_t ColARowB, const double alpha, const double* A, size_t LDimA, const double* B, size_t LDimB, const double beta, double* C, size_t LDimC)
{
   CUBLAS_SAFE_CALL( cublasDgemm(handle, switch_operation_real(opA), switch_operation_real(opB),
                           RowA, ColBC, ColARowB,
                           &alpha,
                           A, LDimA,
                           B, LDimB,
                           &beta,
                           C, LDimC) );  

}
template<> inline
void cublas_wrap::gemm(const char opA, const char opB, size_t RowA, size_t ColBC, size_t ColARowB, const thrust::complex<float> alpha, const thrust::complex<float>* A, size_t LDimA, const thrust::complex<float>* B, size_t LDimB, const thrust::complex<float> beta, thrust::complex<float>* C, size_t LDimC)
{
   CUBLAS_SAFE_CALL( cublasCgemm(handle, switch_operation_complex(opA), switch_operation_complex(opB),
                           RowA, ColBC, ColARowB,
                           (const cuComplex*)&alpha,
                           (const cuComplex*)A, LDimA,
                           (const cuComplex*)B, LDimB,
                           (const cuComplex*)&beta,
                           (cuComplex*)C, LDimC) );  

}
template<> inline
void cublas_wrap::gemm(const char opA, const char opB, size_t RowA, size_t ColBC, size_t ColARowB, const thrust::complex<double> alpha, const thrust::complex<double>* A, size_t LDimA, const thrust::complex<double>* B, size_t LDimB, const thrust::complex<double> beta, thrust::complex<double>* C, size_t LDimC)
{
   CUBLAS_SAFE_CALL( cublasZgemm(handle, switch_operation_complex(opA), switch_operation_complex(opB),
                           RowA, ColBC, ColARowB,
                           (const cuDoubleComplex*)&alpha,
                           (const cuDoubleComplex*)A, LDimA,
                           (const cuDoubleComplex*)B, LDimB,
                           (const cuDoubleComplex*)&beta,
                           (cuDoubleComplex*)C, LDimC) );  

}

#endif