#ifndef __gpu_matrix_vector_operations_H__
#define __gpu_matrix_vector_operations_H__

/*//
 This is a temporal solutoin for BLAS2 operations, when matrices are used as vectors
 These matrices are used in Kryov-type methods like Arnoldi methods.
 TODO: insert SimpleCFD matrices here

*/

#include <cuda_runtime.h>
#include <utils/cuda_support.h>
#include <external_libraries/cublas_wrap.h>
#include <stdexcept>
 

template <typename T, typename T_vec, int BLOCK_SIZE = 1024>
struct gpu_matrix_vector_operations
{
    typedef T  scalar_type;
    typedef T_vec vector_type;
    typedef T* matrix_type;
    bool location;

    //CONSTRUCTORS!
    gpu_matrix_vector_operations(size_t sz_row_, size_t sz_col_, cublas_wrap *cuBLAS_ = nullptr):
    sz_row(sz_row_),
    sz_col(sz_col_),
    cuBLAS(cuBLAS_)
    {
        location=true;
        l_dim_A = sz_row;
    }

    //DISTRUCTOR!
    ~gpu_matrix_vector_operations()
    {
    }

    void set_cublas(cublas_wrap *cuBLAS_p)
    {
        cuBLAS = cuBLAS_p;
    }


    cublas_wrap* get_cublas_ref()
    {
        return cuBLAS;
    }
    void init_matrix(matrix_type& x)const 
    {
        x = NULL;
        //x = device_allocate<scalar_type>(sz);
    } 
    template<class ...Args>
    void init_matrices(Args&&...args) const
    {
        std::initializer_list<int>{((void)init_matrix(std::forward<Args>(args)), 0 )...};
    }
    void start_use_matrix(matrix_type& x)const
    {
        if (x == NULL) 
           x = device_allocate<T>(sz_row*sz_col);
    }  
    template<class ...Args>
    void start_use_matrices(Args&&...args) const
    {
        std::initializer_list<int>{((void)start_use_matrix(std::forward<Args>(args)), 0 )...};
    }      
    void free_matrix(matrix_type& x)const 
    {
        if (x != NULL) 
            cudaFree(x);
    }   
    template<class ...Args>
    void free_matrices(Args&&...args) const
    {
        std::initializer_list<int>{((void)free_matrix(std::forward<Args>(args)), 0 )...};
    }       
    void stop_use_matrix(matrix_type& x)const
    {}
    template<class ...Args>
    void stop_use_matrices(Args&&...args) const
    {
        std::initializer_list<int>{((void)stop_use_matrix(std::forward<Args>(args)), 0 )...};
    }      
    size_t get_rows()
    {
        return sz_row;
    }

    size_t get_cols()
    {
        return sz_col;
    }


    // copies a matrices:  from_ ----> to_
    void assign(const matrix_type from_, matrix_type to_);



    void set_matrix_column(matrix_type mat, const vector_type vec, const size_t col_number);
    void get_matrix_column(vector_type vec, const matrix_type mat,  const size_t col_number);

    void set_matrix_value(matrix_type mat, const scalar_type val, const size_t row_number, const size_t col_number);

    void make_zero_columns(const matrix_type matA, size_t col_from, size_t col_to, matrix_type retA);

    //general GEMV operation
    //void gemv(const char op, size_t RowA, const T *A, size_t ColA, size_t LDimA, const T alpha, const T *x, const T beta, T *y);
    // y <- alpha op(mat)x + beta y
    void gemv(const char op, const matrix_type mat, const scalar_type alpha, const T *x, const scalar_type beta, vector_type y)
    {
        cuBLAS->gemv<scalar_type>(op, sz_row, mat, sz_col, sz_row, alpha, x, beta, y);
    }
    //higher order GEMV operations used in Krylov-type methods

    // dot product of each matrix colums with a vector starting from 0 up till column number max_col-1
    //  vector 'x' should be sized sz_row, 'y' should be the size of max_col at least
    //  y = alpha.*A(:,0:max_col-1)'*x + beta.*y
    void mat2column_dot_vec(const matrix_type mat, size_t max_col, const scalar_type alpha, const T *x, const scalar_type beta, vector_type y)
    {
        if(max_col<=sz_col)
        {
            cuBLAS->gemv<scalar_type>('T', sz_row, mat, max_col, sz_row, alpha, x, beta, y);
        }
        else
        {
            throw std::runtime_error("mat2column_dot_vec: max_col > sz_col");
        }

    }

    // gemv of a matrix that starts from from 0 up till column number max_col-1
    //  vector 'x' should be sized max_col at least, 'y' should be sized sz_row
    //  y = alpha.*A(:,0:max_col-1)*x + beta.*y
    void mat2column_mult_vec(const matrix_type mat, size_t max_col, const scalar_type alpha, const T *x, const scalar_type beta, vector_type y)
    {
        if(max_col<=sz_col)
        {
           
            cuBLAS->gemv<scalar_type>('N', sz_row, mat, max_col, sz_row, alpha, x, beta, y);
        }
        else
        {
            throw std::runtime_error("mat2column_mult_vec: max_col > sz_col");
        }
    }

    //gemm of matrix matrix product. Should not be here, but let's keep it here for a while
    // C = α op ( A ) op ( B ) + β C
    // sizes:
    //      A: sz_row X sz_col
    //      B: sz_col X max_col
    //      C: sz_row X max_col
    // with max_col <= sz_col
    void mat2column_mult_mat(const matrix_type matA, const matrix_type matB, size_t max_col, const scalar_type alpha, const scalar_type beta, matrix_type matC)
    {
        if(max_col<=sz_col)
        {        
            cuBLAS->gemm<scalar_type>('N', 'N', sz_row, max_col, sz_col, alpha, matA, sz_row, matB, sz_col, beta, matC, sz_row);
        }
        else
        {
            throw std::runtime_error("mat2column_mult_mat: max_col > sz_col");            
        }

    }


    // C = α  A^T B + β C
    // A \in R^{NXm}, B \in R^{NXk}, C \in R^{mXk}
    // one can only change the size columns of B and C, i.e. the value of k
    void mat_T_gemm_mat_N(const matrix_type matA, const matrix_type matB, size_t n_cols_B_C, const scalar_type alpha, const scalar_type beta, matrix_type matC)
    {
        cuBLAS->gemm<scalar_type>('T', 'N', sz_col, n_cols_B_C, sz_row , alpha, matA, sz_row, matB, sz_row, beta, matC, sz_col);        
    }

    // C = α op(A) + β B
    void geam(const char opA, size_t RowAC, size_t ColBC, const scalar_type alpha, const scalar_type* A, const scalar_type beta, scalar_type* B, scalar_type* C)
    {
        cuBLAS->geam<scalar_type>(opA, RowAC, ColBC, alpha, A, RowAC,  beta, B, RowAC, C, ColBC);
    }

//*/
private:
    cublas_wrap *cuBLAS;
    size_t sz_row;
    size_t sz_col;
    size_t l_dim_A;//leading_dim_matrix;

    dim3 dimBlock;
    dim3 dimGrid;

};

#endif