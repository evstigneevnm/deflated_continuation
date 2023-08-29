#ifndef __cpu_matrix_vector_operations_var_prec_H__
#define __cpu_matrix_vector_operations_var_prec_H__



#include <stdexcept>
#include <vector> 
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <common/macros.h>


template <class VectorOperations>
struct cpu_matrix_vector_operations_var_prec
{
    
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;
    using matrix_type = std::vector<scalar_type>;
    
    using T = scalar_type;
    using T_vec = vector_type;
    using T_mat = matrix_type;

    //CONSTRUCTORS!
    cpu_matrix_vector_operations_var_prec(size_t sz_row_, size_t sz_col_, const VectorOperations* vec_ops_p):
    sz_rows(sz_row_),
    sz_cols(sz_col_),
    vec_ops_(vec_ops_p)
    {
        location=false;
        l_dim_A = sz_rows;
        vec_ops_->init_vector(vec_helper_row_);
        vec_ops_->start_use_vector(vec_helper_row_);
        vec_ops_->init_vector(vec_helper_col_);
        vec_ops_->start_use_vector(vec_helper_col_);        
    }

    //DISTRUCTOR!
    ~cpu_matrix_vector_operations_var_prec()
    {
        vec_ops_->stop_use_vector(vec_helper_col_);
        vec_ops_->free_vector(vec_helper_col_);
        vec_ops_->stop_use_vector(vec_helper_row_);
        vec_ops_->free_vector(vec_helper_row_);        
    }

    void init_matrix(matrix_type& x)const 
    {

    } 
    template<class ...Args>
    void init_matrices(Args&&...args) const
    {
        std::initializer_list<int>{((void)init_matrix(std::forward<Args>(args)), 0 )...};
    }
    void start_use_matrix(matrix_type& x)const
    {
        if (x.size() == 0) 
           x = std::move( matrix_type(sz_rows*sz_cols, 0) );
    }  
    template<class ...Args>
    void start_use_matrices(Args&&...args) const
    {
        std::initializer_list<int>{((void)start_use_matrix(std::forward<Args>(args)), 0 )...};
    }      
    void free_matrix(matrix_type& x)const 
    {
        if (x.size() > 0) 
            x.resize(0);
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
        return sz_rows;
    }
    size_t get_cols()
    {
        return sz_cols;
    }


    // copies a matrices:  from_ ----> to_
    void assign(const matrix_type& from_, matrix_type& to_)
    {
        std::transform(from_.cbegin(), from_.cend(), to_.begin(), [](T c) { return c; });
    }

    void set_matrix_column(matrix_type& mat, const vector_type& vec, const size_t col_number)const
    {
        if(vec.size() != sz_rows)
        {
            throw std::logic_error("cpu_matrix_vector_operations_var_prec::set_matrix_column: incorrect size of vector and matrix sz_rows");
        }
        for(size_t j=0;j<sz_rows;++j)
        {
            mat[I2_R(j,col_number,sz_rows)] = vec[j];
        }
    }
    void get_matrix_column(vector_type& vec, const matrix_type& mat,  const size_t col_number)const
    {
        if(vec.size() != sz_rows)
        {
            throw std::logic_error("cpu_matrix_vector_operations_var_prec::set_matrix_column: incorrect size of vector and matrix sz_rows");
        }
        for(size_t j=0;j<sz_rows;++j)
        {
            vec[j] = mat[I2_R(j,col_number,sz_rows)];
        }        
    }
    void set_matrix_row(matrix_type& mat, const vector_type& vec, const size_t row_number)const
    {
        if(vec.size() != sz_cols)
        {
            throw std::logic_error("cpu_matrix_vector_operations_var_prec::set_matrix_column: incorrect size of vector and matrix sz_rows");
        }
        for(size_t k=0;k<sz_cols;k++)
        {
            mat[I2_R(row_number,k,sz_rows)] = vec[k];
        }
    }
    void get_matrix_row(vector_type& vec, const matrix_type& mat,  const size_t row_number)const
    {
        if(vec.size() != sz_cols)
        {
            throw std::logic_error("cpu_matrix_vector_operations_var_prec::set_matrix_column: incorrect size of vector and matrix sz_rows");
        }
        for(size_t k=0;k<sz_cols;k++)
        {
            vec[k] = mat[I2_R(row_number,k,sz_rows)];
        }        
    }

    void set_matrix_value(matrix_type& mat, const scalar_type& val, const size_t row_number, const size_t col_number)
    {
        mat[I2_R(row_number,col_number,sz_rows)] = val;
    }

    void assign_random(matrix_type& mat) const
    {
        assign_random(mat, 0.0, 1.0);

    }
    void assign_random(matrix_type& mat, scalar_type a, scalar_type b) const
    {
        #pragma omp parallel for
        for(size_t j=0;j<sz_rows;j++)
        {
            // boost::random::mt19937 gen;
            boost::random::independent_bits_engine<boost::random::mt19937, std::numeric_limits<T>::digits, boost::multiprecision::cpp_int> gen;
            auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            gen.seed( j ); //fix for debug
            boost::random::uniform_real_distribution<T> ur(a, b);
            for(size_t k=0;k<sz_cols;k++)
            {
                mat[I2_R(j,k,sz_rows)] = ur(gen);
            }
        }
    }

    void make_zero_columns(const matrix_type& matA, size_t col_from, size_t col_to, matrix_type& retA)
    {
        for(size_t k = 0; k<sz_cols; ++k)
        {

            if((k >= col_from)&&(k < col_to))
            {
                for(size_t j=0;j<sz_rows;++j)
                {                
                   retA[I2_R(j,k,sz_rows)] = static_cast<scalar_type>(0.0);
               }
            }
            else
            {
                for(size_t j=0;j<sz_rows;++j)
                {
                    retA[I2_R(j,k,sz_rows)] = matA[I2_R(j,k,sz_rows)];
                }            
            }
        }
    }

    //general GEMV operation
    //void gemv(const char op, size_t RowA, const T *A, size_t ColA, size_t LDimA, const T alpha, const T *x, const T beta, T *y);
    // y <- alpha op(mat)x + beta y
    void gemv(const char op, const matrix_type& mat, const scalar_type alpha, const vector_type& x, const scalar_type beta, vector_type& y)
    {
        // cuBLAS->gemv<scalar_type>(op, sz_rows, mat, sz_cols, sz_rows, alpha, x, beta, y);
        
        if(op == 'T')
        {
            #pragma omp parallel for
            for(size_t k = 0;k<sz_cols;k++)
            {
                T val = 0;
                T accum_prod = 0;
                T accum_sum = 0;
                T accum = 0;
                for(size_t j = 0;j<sz_rows;j++)
                {
                    T prod = two_prod(accum_prod, x[j], mat[I2_R(j,k,sz_rows)]);
                    val = two_sum(accum_sum, val, prod );
                    accum = accum + accum_prod + accum_sum;
                    // val = val + x[k]*mat[I2_R(j,k,sz_rows)];
                }
                y[k] = val+accum;                

            }
        }
        else
        {
            #pragma omp parallel for
            for(size_t j = 0;j<sz_rows;j++)
            {
                // get_matrix_row(vec_helper_row_, mat,  j);
                // y[j] = vec_ops_->scalar_prod(vec_helper_row_, x);
                T val = 0;
                T accum_prod = 0;
                T accum_sum = 0;
                T accum = 0;
                for(size_t k = 0;k<sz_cols;k++)
                {
                    T prod = two_prod(accum_prod, x[k], mat[I2_R(j,k,sz_rows)]);
                    val = two_sum(accum_sum, val, prod );
                    accum = accum + accum_prod + accum_sum;
                    // val = val + x[k]*mat[I2_R(j,k,sz_rows)];
                }
                y[j] = val+accum;
            }
        }

    }
    //higher order GEMV operations used in Krylov-type methods

    // dot product of each matrix colums with a vector starting from 0 up till column number max_col-1
    //  vector 'x' should be sized sz_rows, 'y' should be the size of max_col at least
    //  y = alpha.*A(:,0:max_col-1)'*x + beta.*y
    void mat2column_dot_vec(const matrix_type& mat, size_t max_col, const scalar_type alpha, const vector_type& x, const scalar_type beta, vector_type& y)
    {
        if(max_col<=sz_cols)
        {
            #pragma omp parallel for
            for(size_t k = 0;k<max_col;k++)
            {
                T val = 0;
                T accum_prod = 0;
                T accum_sum = 0;
                T accum = 0;
                for(size_t j = 0;j<sz_rows;j++)
                {
                    T prod = two_prod(accum_prod, x[j], mat[I2_R(j,k,sz_rows)]);
                    val = two_sum(accum_sum, val, prod );
                    accum = accum + accum_prod + accum_sum;
                    // val = val + x[k]*mat[I2_R(j,k,sz_rows)];
                }
                y[k] = val+accum;                

            }
        }
        else
        {
            throw std::runtime_error("mat2column_dot_vec: max_col > sz_cols");
        }

    }

    // gemv of a matrix that starts from from 0 up till column number max_col-1
    //  vector 'x' should be sized max_col at least, 'y' should be sized sz_rows
     // y = alpha.*A(:,0:max_col-1)*x + beta.*y
    void mat2column_mult_vec(const matrix_type& mat, size_t max_col, const scalar_type alpha, const vector_type& x, const scalar_type beta, vector_type& y)
    {
        if(max_col<=sz_cols)
        {
            #pragma omp parallel for
            for(size_t k = 0;k<max_col;k++)
            {
                T val = 0;
                T accum_prod = 0;
                T accum_sum = 0;
                T accum = 0;
                for(size_t j = 0;j<sz_rows;j++)
                {
                    T prod = two_prod(accum_prod, x[k], mat[I2_R(j,k,sz_rows)]);
                    val = two_sum(accum_sum, val, prod );
                    accum = accum + accum_prod + accum_sum;
                    // val = val + x[k]*mat[I2_R(j,k,sz_rows)];
                }
                y[k] = val+accum;                

            }           
            
        }
        else
        {
            throw std::runtime_error("mat2column_mult_vec: max_col > sz_cols");
        }
    }

    //gemm of matrix matrix product. Should not be here, but let's keep it here for a while
    // C = α op ( A ) op ( B ) + β C
    // sizes:
    //      A: sz_rows X sz_cols
    //      B: sz_cols X max_col
    //      C: sz_rows X max_col
    // with max_col <= sz_cols
    // void gemm(const char opA, const chat opB, )
    // {

    // }

    void mat2column_mult_mat(const matrix_type& matA, const matrix_type& matB, size_t max_col, const scalar_type alpha, const scalar_type beta, matrix_type& matC)
    {
        if(max_col<=sz_cols)
        {        
            
        }
        else
        {
            throw std::runtime_error("mat2column_mult_mat: max_col > sz_cols");            
        }

    }


    // C = α  A^T B + β C
    // A \in R^{NXm}, B \in R^{NXk}, C \in R^{mXk}
    // one can only change the size columns of B and C, i.e. the value of k
    // void mat_T_gemm_mat_N(const matrix_type matA, const matrix_type matB, size_t n_cols_B_C, const scalar_type alpha, const scalar_type beta, matrix_type matC)
    // {
    //     cuBLAS->gemm<scalar_type>('T', 'N', sz_cols, n_cols_B_C, sz_rows , alpha, matA, sz_rows, matB, sz_rows, beta, matC, sz_cols);        
    // }

    // C = α op(A) + β B
    void geam(const char opA, size_t RowAC, size_t ColBC, const scalar_type alpha, const matrix_type& A, const scalar_type beta, matrix_type& B, matrix_type& C)
    {
        // cuBLAS->geam<scalar_type>(opA, RowAC, ColBC, alpha, A, RowAC,  beta, B, RowAC, C, ColBC);
        if(opA == 'T')
        {
            #pragma omp parallel for
            for(size_t k = 0;k<ColBC;k++)
            {
                #pragma omp parallel for
                for(size_t j = 0;j<RowAC;j++)
                {
                    C[I2_R(j,k,RowAC)] = alpha*A[I2_R(k,j,RowAC)] + beta*B[I2_R(j,k,RowAC)];
                }
            }
        }
        else
        {
            #pragma omp parallel for
            for(size_t k = 0;k<ColBC;k++)
            {
                #pragma omp parallel for
                for(size_t j = 0;j<RowAC;j++)
                {
                    C[I2_R(j,k,RowAC)] = alpha*A[I2_R(j,k,RowAC)] + beta*B[I2_R(j,k,RowAC)];
                }
            }
        }
    }


    [[nodiscard]] T norm_fro(const matrix_type& A) const
    { 
        int omp_num_threads = 0;
        #pragma omp parallel
        {
            omp_num_threads = omp_get_num_threads();
        }
        T res = 0.0;
        std::vector<T> res_th(omp_num_threads, 0.0);

        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            #pragma omp for
            for (size_t j = 0; j < sz_cols; ++j)
            {
                for (size_t k = 0; k < sz_rows; ++k)
                {
                    auto a_jk = A[I2_R(j,k,sz_rows)];   
                    res_th[thread_id] = res_th[thread_id] + a_jk*a_jk;
                }
            }            
        }
        for(int l = 0; l<omp_num_threads;++l)
        {
            res = res + res_th[l];
        }
        
    

        // for(size_t k = 0;k<sz_cols;k++)
        // {
        //     for(size_t j = 0;j<sz_rows;j++)
        //     {
        //         auto a_jk = A[I2_R(j,k,sz_rows)];
        //         res = res + a_jk*a_jk;
        //     }
        // }
        return boost::multiprecision::sqrt(res);
    }

    //gemm of matrix matrix product. Should not be here, but let's keep it here for a while
    // C = α op ( A ) op ( B ) + β C
    void gemm(const char opA, const char opB, size_t sz_row_A_row_C, size_t sz_col_A_row_B, size_t max_col_B_cols_C, const T alpha, const matrix_type& matA, const matrix_type& matB, const T beta, matrix_type& matC)
    {



        for(size_t j = 0;j<sz_row_A_row_C;j++)
        {

            for(size_t k = 0;k<max_col_B_cols_C;k++)
            {
                
                for(size_t l = 0;l<sz_col_A_row_B)
                {
                    

                }

            }
        }
    

    }




//*/
private:


    T two_prod(T &t, T a, T b) const // [1], pdf: 71, 169, 198, 
    {
        T p = a*b;
        t = boost::multiprecision::fma(a, b, -p);
        return p;
    }

    T two_sum(T &t, T a, T b) const
    {
        T s = a+b;
        T bs = s-a;
        T as = s-bs;
        t = (b-bs) + (a-as);
        return s;
    }


    size_t sz_rows;
    size_t sz_cols;
    size_t l_dim_A;//leading_dim_matrix;
    bool location;
    mutable vector_type vec_helper_row_;
    mutable vector_type vec_helper_col_;
    const VectorOperations* vec_ops_;

};

#endif