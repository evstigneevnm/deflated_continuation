/*
 *     This file is part of Common_GPU_Operations.
 *     Copyright (C) 2009-2021  Evstigneev Nikolay Mikhaylovitch <evstigneevnm@ya.ru>, Ryabkov Oleg Igorevitch
 *
 *     This program is free software: you can redistribute it and/or modify
 *     it under the terms of the GNU General Public License as published by
 *     the Free Software Foundation, either version 3 of the License, or
 *     (at your option) any later version.
 *
 *     This program is distributed in the hope that it will be useful,
 *     but WITHOUT ANY WARRANTY; without even the implied warranty of
 *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PUPOSE.  See the
 *     GNU General Public License for more details.
 *
 *     You should have received a copy of the GNU General Public License
 *     along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *      */

#ifndef __CUSOLVER_WRAP_H__
#define __CUSOLVER_WRAP_H__

#include <iostream>
#include <cuda_runtime.h>
#include <cusolverDn.h>
// #include <thrust/complex.h>
#include <external_libraries/cublas_wrap.h>
#include <utils/cusolver_safe_call.h>
#include <utils/cuda_safe_call.h>
#include <stdexcept>

class cusolver_wrap
{
    
//  used for the solution of the linear system
    using blas_t = cublas_wrap;
    
//  simple matrix structure that is RAII
    template<class T>
    struct _A_t
    {
        mutable T* data_ = nullptr;
        mutable size_t sz_ = 0;
        ~_A_t()
        {
            if(data_ != nullptr)
            {
                cudaFree(data_);
            }
        }
        void init(size_t rows_, size_t cols_, const T* A) const
        {
            if(data_ != nullptr)
            {
                cudaFree(data_);
                data_=nullptr;
            }
            sz_ = rows_*cols_;
            CUDA_SAFE_CALL(cudaMalloc((void**)&data_, sizeof(T)*sz_) ); 
            copy(A);
        }
        size_t get_rowcols()
        {
            return sz_;
        }
        void copy(const T* A) const
        {
            CUDA_SAFE_CALL( cudaMemcpy(data_, A, sizeof(T)*sz_, cudaMemcpyDeviceToDevice) );            
        }

    };


public:
    
    cusolver_wrap(): handle_created(false)
    {
        cusolver_create();
        handle_created=true;
    }
    
    cusolver_wrap(blas_t* cublas_): 
    handle_created(false)
    {
        cusolver_create();
        set_cublas(cublas_);
        handle_created=true;
    }

    cusolver_wrap(bool plot_info): handle_created(false)
    {
        if(plot_info)
        {
            cusolver_create_info();
            handle_created=true;
        }
        else
        {
            cusolver_create();
            handle_created=true;
        }
    }


    ~cusolver_wrap()
    {
        free_d_work_double();
        free_d_work_float();
        if(handle_created)
        {
            cusolver_destroy();
            handle_created=false;
        }

      
    }
 
    cusolverDnHandle_t* get_handle()
    {
        return &handle;
    }
    
    template<typename T> //only symmetric matrix is computed, only real eigenvalues
    void eig(size_t rows_cols, T* A, T* lambda); //returns matrix of left eigs in A



    void set_cublas(blas_t* cublas_)
    {
        if(!cublas_set)
        {
            cublas = cublas_;
        }
        cublas_set = true;
        // printf("set_cublas: address of cublas is %p\n", (void *)cublas ); 
    }

//  WARNING!
//  1. matrix A WILL BE OVERWRITTEN
//  2. matrix must be Column-major order: A_{j,k} = data[rows*k+j];
    template<typename T>
    void gesv(const size_t rows_cols, T* A, T* b_x)
    {
        check_blas();
        qr_size('T',
                'L',
                rows_cols,
                rows_cols,
                A,
                rows_cols,
                b_x,
                rows_cols
                );
        geqrf_ormqr(
                'T',
                'L',
                rows_cols,
                rows_cols,
                A,
                rows_cols,
                b_x,
                rows_cols
                );
        
        cublas->trsm('L', 'U', 'N', false, rows_cols, 1, T(1.0), A, rows_cols, b_x, rows_cols);

    } 

    template<typename T>
    void gesv(blas_t* cublas_, const size_t rows_cols, T* A, T* b_x)
    {
        set_cublas(cublas_);
        gesv(rows_cols, A, b_x);
    }

    

    template<typename T>
    void gesv(const size_t rows_cols, const T* A, const T* b, T* x)
    {
        check_blas();
        _A_t<T> _A_;
        _A_.init(rows_cols, rows_cols, A);
        CUDA_SAFE_CALL( cudaMemcpy(x, b, sizeof(T)*rows_cols, cudaMemcpyDeviceToDevice) );
        gesv(rows_cols, _A_.data_, x);
    }

    template<typename T>
    void gesv(blas_t* cublas_, const size_t rows_cols, const T* A, const T* b, T* x)
    {
        set_cublas(cublas_);
        gesv(rows_cols, A, b, x);
    }


private:

    bool handle_created = false;
    cusolverDnHandle_t handle;        
    double* d_work_d = nullptr;
    float* d_work_f = nullptr;
    int work_size = 0;
    blas_t* cublas;
    bool cublas_set = false;

    float* tau_f = nullptr; //elementary reflections vector
    double* tau_d = nullptr; //elementary reflections vector
    int tau_size = 0;


    void check_blas()
    {
        if(!cublas_set)
        {
            throw std::logic_error("cusolver_wrap::check_blas: cublas handle is not set.");
        }
    }

    void free_tau_d()
    {
        if(tau_d!=nullptr)
        {
            cudaFree(tau_d);
            tau_d = nullptr;
        }
    }
    void free_tau_f()
    {
        if(tau_f!=nullptr)
        {
            cudaFree(tau_f);
            tau_f = nullptr;
        }
    }    
    void set_tau_double(int tau_size_)
    {
        if(tau_size<tau_size_)
        {
            free_tau_d();
        }
        tau_size = tau_size_;
        CUDA_SAFE_CALL(cudaMalloc((void**)&tau_d, sizeof(double)*tau_size) ); 

    }
    void set_tau_float(int tau_size_)
    {
        if(tau_size<tau_size_)
        {
            free_tau_f();
        }
        tau_size = tau_size_;
        CUDA_SAFE_CALL(cudaMalloc((void**)&tau_f, sizeof(float)*tau_size) ); 
    }

    template<class T>
    void qr_size(
                char operation,
                char side,
                size_t m,
                size_t n,
                const T *A,
                size_t lda,
                const T *b,
                size_t ldb
                );

    template<class T>
    void geqrf_ormqr(
                char operation,
                char side,
                size_t m,
                size_t n,
                T *A,
                size_t lda,
                T *b,
                size_t ldb
                );

    void cusolver_destroy()
    {
        CUSOLVER_SAFE_CALL(cusolverDnDestroy(handle));
    }
    
    void cusolver_create()
    {
        CUSOLVER_SAFE_CALL(cusolverDnCreate(&handle));
    }

    void cusolver_create_info()
    {
        cusolver_create();
        int cusolver_version;
        int major_ver, minor_ver, patch_level;
        CUSOLVER_SAFE_CALL(cusolverGetVersion(&cusolver_version));
        CUSOLVER_SAFE_CALL(cusolverGetProperty(MAJOR_VERSION, &major_ver));
        CUSOLVER_SAFE_CALL(cusolverGetProperty(MINOR_VERSION, &minor_ver));
        CUSOLVER_SAFE_CALL(cusolverGetProperty(PATCH_LEVEL, &patch_level));
        std::cout << "cuSOLVER v."<< cusolver_version << " (major="<< major_ver << ", minor=" << minor_ver << ", patch level=" << patch_level << ") handle created." << std::endl;
    }

    void free_d_work_double()
    {
        if(d_work_d!=nullptr)
        {
            cudaFree(d_work_d);
            d_work_d = nullptr;
        }        
    }
    void free_d_work_float()
    {
        if(d_work_f!=nullptr)
        {
            cudaFree(d_work_f);
            d_work_f = nullptr;
        }        
    }
    void set_d_work_double(int work_size_)
    {
        if(work_size<work_size_)
        {
            work_size = work_size_;
            free_d_work_double();
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_work_d, sizeof(double)*work_size) );     
        }
    }
    void set_d_work_float(int work_size_)
    {
        if(work_size<work_size_)
        {
            work_size = work_size_;
            free_d_work_float();
            CUDA_SAFE_CALL(cudaMalloc((void**)&d_work_f, sizeof(float)*work_size) );     
        }
    }

};



template<> inline
void cusolver_wrap::geqrf_ormqr(char operation_, char side_, size_t rows, size_t cols, double *A, size_t lda, double* b, size_t ldb)
{
    int *devInfo = nullptr;
    CUDA_SAFE_CALL(cudaMalloc ((void**)&devInfo, sizeof(int)) );
    int info_gpu;
    CUSOLVER_SAFE_CALL
    (
        cusolverDnDgeqrf
        (
            handle,
            (int) rows,
            (int) cols, 
            A, 
            lda, 
            tau_d, 
            d_work_d, 
            work_size, 
            devInfo
        )
    );
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    CUDA_SAFE_CALL( cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
    if(info_gpu != 0)
    {
        throw std::runtime_error("cusolver_wrap::geqrf_ormqr.geqrf: info_gpu = " + std::to_string(info_gpu) );
    }
    int m = rows;
    int n = 1;
    int k = rows;
    cublasSideMode_t side = CUBLAS_SIDE_LEFT;
    if((side_ == 'r')||(side_ == 'R'))
    {
        side = CUBLAS_SIDE_RIGHT;
        m = 1;
        n = cols;
    }
    cublasOperation_t trans = CUBLAS_OP_N;
    if((operation_ == 't')||(operation_ == 'T')) 
    {
        trans = CUBLAS_OP_T;
    }  

    CUSOLVER_SAFE_CALL
    (    
        cusolverDnDormqr
        (
            handle,
            side,
            trans,
            m,
            n,
            k,
            A,
            lda,
            tau_d,
            b,
            ldb,
            d_work_d,
            work_size,
            devInfo
        )
    );
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    CUDA_SAFE_CALL( cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
    if(info_gpu != 0)
    {
        throw std::runtime_error("cusolver_wrap::geqrf_ormqr.ormqr: info_gpu = " + std::to_string(info_gpu) );
    }    

}
template<> inline
void cusolver_wrap::geqrf_ormqr(char operation_, char side_, size_t rows, size_t cols, float *A, size_t lda, float* b, size_t ldb)
{
    int *devInfo = nullptr;
    CUDA_SAFE_CALL(cudaMalloc ((void**)&devInfo, sizeof(int)) );
    int info_gpu;
    CUSOLVER_SAFE_CALL
    (
        cusolverDnSgeqrf
        (
            handle,
            (int) rows,
            (int) cols, 
            A, 
            lda, 
            tau_f, 
            d_work_f, 
            work_size, 
            devInfo
        )
    );
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    CUDA_SAFE_CALL( cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
    if(info_gpu != 0)
    {
        throw std::runtime_error("cusolver_wrap::geqrf_ormqr.geqrf: info_gpu = " + std::to_string(info_gpu) );
    }
    int m = rows;
    int n = 1;
    int k = rows;
    cublasSideMode_t side = CUBLAS_SIDE_LEFT;
    if((side_ == 'r')||(side_ == 'R'))
    {
        side = CUBLAS_SIDE_RIGHT;
        m = 1;
        n = cols;
    }
    cublasOperation_t trans = CUBLAS_OP_N;
    if((operation_ == 't')||(operation_ == 'T')) 
    {
        trans = CUBLAS_OP_T;
    }  

    CUSOLVER_SAFE_CALL
    (    
        cusolverDnSormqr
        (
            handle,
            side,
            trans,
            m,
            n,
            k,
            A,
            lda,
            tau_f,
            b,
            ldb,
            d_work_f,
            work_size,
            devInfo
        )
    );
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    CUDA_SAFE_CALL( cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
    if(info_gpu != 0)
    {
        throw std::runtime_error("cusolver_wrap::geqrf_ormqr.ormqr: info_gpu = " + std::to_string(info_gpu) );
    }    

}

template<> inline
void cusolver_wrap::qr_size(char operation_, char side_, size_t rows, size_t cols, const double *A, size_t lda, const double* b, size_t ldb)
{
    
    int lwork_1 = 0;
    int lwork_2 = 0;
    CUSOLVER_SAFE_CALL
    (
        cusolverDnDgeqrf_bufferSize
        (
            handle,
            (int) rows,
            (int) cols,
            (double*)A,
            (int) lda,
            &lwork_1
        )
    );
    
    int m = rows;
    int n = 1;
    int k = rows;
    cublasSideMode_t side = CUBLAS_SIDE_LEFT;
    if((side_ == 'r')||(side_ == 'R'))
    {
        side = CUBLAS_SIDE_RIGHT;
        m = 1;
        n = cols;
    }
    cublasOperation_t trans = CUBLAS_OP_N;
    if((operation_ == 't')||(operation_ == 'T')) 
    {
        trans = CUBLAS_OP_T;
    }

    set_tau_double(int(rows));
    CUSOLVER_SAFE_CALL
    (    
        cusolverDnDormqr_bufferSize
        (
            handle,
            side,
            trans,
            m,
            n,
            k,
            A,
            lda,
            tau_d,
            b,
            (int)ldb,
            &lwork_2
        )
    );

    int lwork = (lwork_1>lwork_2)?lwork_1:lwork_2;
    set_d_work_double(lwork);
}

template<> inline
void cusolver_wrap::qr_size(char operation_, char side_, size_t rows, size_t cols, const float *A, size_t lda, const float* b, size_t ldb)
{
    
    int lwork_1 = 0;
    int lwork_2 = 0;
    CUSOLVER_SAFE_CALL
    (
        cusolverDnSgeqrf_bufferSize
        (
            handle,
            (int) rows,
            (int) cols,
            (float*)A,
            (int) lda,
            &lwork_1
        )
    );
    
    int m = rows;
    int n = 1;
    int k = rows;
    cublasSideMode_t side = CUBLAS_SIDE_LEFT;
    if((side_ == 'r')||(side_ == 'R'))
    {
        side = CUBLAS_SIDE_RIGHT;
        m = 1;
        n = cols;
    }
    cublasOperation_t trans = CUBLAS_OP_N;
    if((operation_ == 't')||(operation_ == 'T')) 
    {
        trans = CUBLAS_OP_T;
    }

    set_tau_float(int(rows));
    CUSOLVER_SAFE_CALL
    (    
        cusolverDnSormqr_bufferSize
        (
            handle,
            side,
            trans,
            m,
            n,
            k,
            A,
            lda,
            tau_f,
            b,
            (int)ldb,
            &lwork_2
        )
    );

    int lwork = (lwork_1>lwork_2)?lwork_1:lwork_2;
    set_d_work_float(lwork);
}



template<> inline
void cusolver_wrap::eig(size_t rows_cols, double* A, double* lambda)
{
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    int m = rows_cols;
    int lda = m;
    int lwork = 0;
    
    int *devInfo = nullptr;
    CUDA_SAFE_CALL(cudaMalloc ((void**)&devInfo, sizeof(int)) );
    int info_gpu;

    CUSOLVER_SAFE_CALL
    (
        cusolverDnDsyevd_bufferSize
        (
            handle,
            jobz,
            uplo,
            m,
            A,
            lda,
            lambda,
            &lwork
        )
    );
    set_d_work_double(lwork);

    CUSOLVER_SAFE_CALL
    (
        cusolverDnDsyevd
        (
            handle,
            jobz,
            uplo,
            m,
            A,
            lda,
            lambda,
            d_work_d,
            lwork,
            devInfo
        )
    );
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    CUDA_SAFE_CALL( cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
    if(info_gpu!=0)
    {
        throw std::runtime_error("cusolver_wrap::eig: info_gpu = " + std::to_string(info_gpu) );
    }
}

template<> inline
void cusolver_wrap::eig(size_t rows_cols, float* A, float* lambda)
{
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
    int m = rows_cols;
    int lda = m;
    int lwork = 0;
    
    int *devInfo = nullptr;
    CUDA_SAFE_CALL(cudaMalloc ((void**)&devInfo, sizeof(int)) );
    int info_gpu;

    CUSOLVER_SAFE_CALL
    (
        cusolverDnSsyevd_bufferSize
        (
            handle,
            jobz,
            uplo,
            m,
            A,
            lda,
            lambda,
            &lwork
        )
    );
    set_d_work_float(lwork);

    CUSOLVER_SAFE_CALL
    (
        cusolverDnSsyevd
        (
            handle,
            jobz,
            uplo,
            m,
            A,
            lda,
            lambda,
            d_work_f,
            lwork,
            devInfo
        )
    );
    CUDA_SAFE_CALL( cudaDeviceSynchronize() );
    CUDA_SAFE_CALL( cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
    if(info_gpu!=0)
    {
        throw std::runtime_error("cusolver_wrap::eig: info_gpu = " + std::to_string(info_gpu) );
    }

}


#endif
