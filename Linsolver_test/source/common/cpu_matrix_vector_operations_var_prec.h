#ifndef __cpu_matrix_vector_operations_var_prec_H__
#define __cpu_matrix_vector_operations_var_prec_H__


#include <cmath>
#include <random>
#include <stdexcept>
#include <vector> 
#include <algorithm>
#include <chrono>
#include <omp.h>
#include <common/macros.h>

#if defined(BOOST_MP_CPP_BIN_FLOAT_HPP)||defined(BOOST_MP_CPP_INT_HPP)
#define __cpu_matrix_vector_operations_var_prec_H_use_boost__
#endif

template <class VectorOperations>
struct cpu_matrix_vector_operations_var_prec
{
private:
    #ifdef __cpu_matrix_vector_operations_var_prec_H_use_boost__
        template<class T>
        auto sqrt(T val)const {return boost::multiprecision::sqrt(val);}
        template<class T>
        auto abs(T val)const {return boost::multiprecision::abs(val);}
        template<class T>
        auto fma(T a, T b, T c)const{return boost::multiprecision::fma(a,b,c);}


    #else
        template<class T>
        auto sqrt(T val)const {return std::sqrt(val);}
        template<class T>
        auto abs(T val)const {return std::abs(val);}     
        template<class T>
        auto fma(T a, T b, T c)const {return std::fma(a,b,c);}

    #endif



public:
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
        machesp_ = macheps();
        std::cout << "cpu_matrix_vector_operations_var_prec: macheps = " << machesp_ << std::endl;
        init_matrix(helper_matrix_);
        start_use_matrix(helper_matrix_);
    }

    //DISTRUCTOR!
    ~cpu_matrix_vector_operations_var_prec()
    {
        stop_use_matrix(helper_matrix_);
        free_matrix(helper_matrix_);
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
    size_t get_rows() const
    {
        return sz_rows;
    }
    size_t get_cols() const
    {
        return sz_cols;
    }


    // copies a matrices:  from_ ----> to_
    void assign(const matrix_type& from_, matrix_type& to_)const
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

    void set_matrix_value(matrix_type& mat, const scalar_type& val, const size_t row_number, const size_t col_number) const
    {
        mat[I2_R(row_number,col_number,sz_rows)] = val;
    }

    void assign_random(matrix_type& mat) const
    {
        assign_random(mat, static_cast<scalar_type>(0.0), static_cast<scalar_type>(1.0) );

    }

    #ifdef __cpu_matrix_vector_operations_var_prec_H_use_boost__
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
    #else
    void assign_random(matrix_type& mat, scalar_type a, scalar_type b) const
    {
        #pragma omp parallel for
        for(size_t j=0;j<sz_rows;j++)
        {
            
            std::independent_bits_engine<std::mt19937_64, std::numeric_limits<T>::digits, uint64_t> gen;
            auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            gen.seed( j ); //fix for debug
            std::uniform_real_distribution<T> ur(a, b);
            for(size_t k=0;k<sz_cols;k++)
            {
                mat[I2_R(j,k,sz_rows)] = ur(gen);
            }
        }
    }
    #endif

    void make_zero_columns(const matrix_type& matA, size_t col_from, size_t col_to, matrix_type& retA) const
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
    // y <- alpha op(A)x + beta y
    void gemv(const char op, const matrix_type& mat, const scalar_type alpha, const vector_type& x, const scalar_type beta, vector_type& y) const
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
                y[k] = alpha*(val+accum) + beta*y[k];                

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
                y[j] = alpha*(val+accum) + beta*y[j];
            }
        }

    }
    //higher order GEMV operations used in Krylov-type methods

    // dot product of each matrix colums with a vector starting from 0 up till column number max_col-1
    //  vector 'x' should be sized sz_rows, 'y' should be the size of max_col at least
    //  y = alpha.*A(:,0:max_col-1)'*x + beta.*y
    void mat2column_dot_vec(const matrix_type& mat, size_t max_col, const scalar_type alpha, const vector_type& x, const scalar_type beta, vector_type& y) const
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
    void mat2column_mult_vec(const matrix_type& mat, size_t max_col, const scalar_type alpha, const vector_type& x, const scalar_type beta, vector_type& y) const
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




    // C = α op(A) + β B
    void geam(const char opA, size_t RowAC, size_t ColBC, const scalar_type alpha, const matrix_type& A, const scalar_type beta, matrix_type& B, matrix_type& C) const
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
        T res = static_cast<scalar_type>(0.0);
        std::vector<T> res_th(omp_num_threads, static_cast<scalar_type>(0.0) );

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
        return sqrt(res);
    }

    //gemm of matrix matrix product. Should not be here, but let's keep it here for a while
    // C = α op ( A ) op ( B ) + β C
    // sizes:
    //      A: sz_rows X sz_cols
    //      B: sz_cols X max_col
    //      C: sz_rows X max_col
    // with max_col <= sz_cols
    //gemm of matrix matrix product. Should not be here, but let's keep it here for a while
    // C = α op ( A ) op ( B ) + β C
    // C_{j,k} = \sum_{l=0}^{colsA_rowsB} A_{j,l}*B_{l,k}, for j=1,..,colsA, k=1,..,rowsB
    void gemm(const char opA, const char opB, const T alpha, size_t sz_row_A, size_t sz_col_A, const matrix_type& matA, size_t sz_row_B, size_t sz_col_B, size_t max_size_B, const matrix_type& matB, const T beta, matrix_type& matC) const
    {



        if((opA == 'N')&&(opB == 'N'))
        {
            if(sz_col_A!=sz_row_B)
            {
                throw std::logic_error("gemm: matrix sizes are invalid: A(" + std::to_string(sz_row_A)+"X"+std::to_string(sz_col_A)+") * B("+ std::to_string(sz_row_B)+"X"+std::to_string(sz_col_B)+")");
            }
            #pragma omp parallel for
            for(size_t j = 0;j<sz_row_A;j++)
            {

                for(size_t k = 0;k<max_size_B;k++)
                {
                    
                    matC[I2_R(j,k, sz_row_A)] *= beta;
                    for(size_t l = 0;l<sz_col_A;l++)
                    {
                        
                        matC[I2_R(j,k, sz_row_A)] += alpha*matA[I2_R(j,l, sz_row_A)]*matB[I2_R(l,k, sz_row_B)];
                    }

                }
            }
        }
        else if((opA == 'T')&&(opB == 'N'))
        {
            if(sz_row_A!=sz_row_B)
            {
                throw std::logic_error("gemm: matrix sizes are invalid: A(" + std::to_string(sz_row_A)+"X"+std::to_string(sz_col_A)+")^T * B("+ std::to_string(sz_row_B)+"X"+std::to_string(sz_col_B)+")");
            }            
            #pragma omp parallel for
            for(size_t j = 0;j<sz_col_A;j++)
            {

                for(size_t k = 0;k<max_size_B;k++)
                {
                    
                    matC[I2_R(j,k, sz_row_A)] *= beta;
                    for(size_t l = 0;l<sz_row_A;l++)
                    {
                        
                        matC[I2_R(j,k, sz_row_A)] += alpha*matA[I2_R(l,j, sz_row_A)]*matB[I2_R(l,k, sz_row_B)];
                    }

                }
            }
        }    
        else if((opA == 'N')&&(opB == 'T'))
        {
            if(sz_col_A!=sz_col_B)
            {
                throw std::logic_error("gemm: matrix sizes are invalid: A(" + std::to_string(sz_row_A)+"X"+std::to_string(sz_col_A)+") * B("+ std::to_string(sz_row_B)+"X"+std::to_string(sz_col_B)+")^T");
            }            
            #pragma omp parallel for
            for(size_t j = 0;j<sz_row_A;j++)
            {

                for(size_t k = 0;k<max_size_B;k++)
                {
                    
                    matC[I2_R(j,k, sz_row_A)] *= beta;
                    for(size_t l = 0;l<sz_col_A;l++)
                    {
                        
                        matC[I2_R(j,k, sz_row_A)] += alpha*matA[I2_R(j,l, sz_row_A)]*matB[I2_R(k,l, sz_row_B)];
                    }

                }
            }
        }
        else
        {
            throw std::logic_error("gemm: operations A^T * B^T not spported.");
        }

    }


    // C = α  A^T B + β C
    // A \in R^{NXm}, B \in R^{NXk}, C \in R^{mXk}
    // one can only change the size columns of B and C, i.e. the value of k
    void mat_T_gemm_mat_N(const matrix_type& matA, const matrix_type& matB, size_t n_cols_B_C, const scalar_type alpha, const scalar_type beta, matrix_type& matC) const
    {
        if(n_cols_B_C<=sz_cols)
        { 
            gemm('T', 'N', alpha, sz_rows, sz_cols, matA, sz_rows, sz_cols, n_cols_B_C, matB, beta, matC);        
        }
        else
        {
            throw std::runtime_error("mat_T_gemm_mat_N: n_cols_B_C > sz_cols"); 
        }
    }

    //gemm of matrix matrix product. Should not be here, but let's keep it here for a while
    // C = α op ( A ) op ( B ) + β C
    // sizes:
    //      A: sz_row X sz_col
    //      B: sz_col X max_col
    //      C: sz_row X max_col
    // with max_col <= sz_col
    void mat2column_mult_mat(const matrix_type& matA, const matrix_type& matB, size_t max_col, const scalar_type alpha, const scalar_type beta, matrix_type& matC) const
    {
        if(max_col<=sz_cols)
        {        
            gemm('N', 'N', alpha, sz_rows, sz_cols, matA, sz_rows, sz_cols, max_col, matB, beta, matC);  
        }
        else
        {
            throw std::runtime_error("mat2column_mult_mat: max_col > sz_cols");            
        }

    }


    void lup_decomposition(matrix_type& A, matrix_type& P) const
    {
        lup_decomp_s lup(A, sz_rows, sz_cols, machesp_);
        lup.permutation_matrix(P);
    }
    void gesv(matrix_type& A, const vector_type&b, vector_type&x)const 
    {
        lup_decomp_s lup(A, sz_rows, sz_cols, machesp_);
        lup.solve(A, b, x);
    }
 
    void gesv(matrix_type& A, vector_type&x)const 
    {
        lup_decomp_s lup(A, sz_rows, sz_cols, machesp_);
        vec_ops_->assign(x, vec_helper_row_);
        lup.solve(A, vec_helper_row_, x);
    }    

    void gesv(const matrix_type& A, const vector_type&b, vector_type&x)const
    {
        assign(A, helper_matrix_);
        lup_decomp_s lup(helper_matrix_, sz_rows, sz_cols, machesp_);
        lup.solve(helper_matrix_, b, x);
    }

    void gesv(const size_t rows_cols, matrix_type& A, vector_type& b_x)const
    {
        if(rows_cols == sz_rows)
        {
            gesv(A, b_x);
        }
        else
        {
            throw std::runtime_error("gesv: incorrect matrix dims");
        }
    }
    void gesv(const size_t rows_cols, const matrix_type& A, const vector_type& b, vector_type& x) const
    {
        if(rows_cols == sz_rows)
        {
            gesv(A, b, x);
        }
        else
        {
            throw std::runtime_error("gesv: incorrect matrix dims");
        }        
    }

    void inv(matrix_type& A, matrix_type& iA)const
    {
        lup_decomp_s lup(A, sz_rows, sz_cols, machesp_);
        lup.inv(A, iA);
    }
    T det(matrix_type& A) const
    {
        lup_decomp_s lup(A, sz_rows, sz_cols, machesp_);
        return lup.det(A);
    }

//*/
private:
    T machesp_;
    T macheps()
    {
        T a = static_cast<scalar_type>(1.0);
        while(a + static_cast<scalar_type>(1.0) != static_cast<scalar_type>(1.0) )
        {
            a = static_cast<T>(0.5)*a;
        }
        return a;
    }



    struct lup_decomp_s
    {
        lup_decomp_s(matrix_type& A, size_t sz_rows, size_t sz_cols, const T macheps = 1.0e-12):
        sz_rows_(sz_rows),
        sz_cols_(sz_cols),
        machesp_(macheps)
        {
            if(sz_rows_!=sz_cols_)
            {
                wrong_size_ = true;
            }
            else
            {
                wrong_size_ = false;
                P_ = std::vector<size_t>(sz_rows_+1);
            }
            lup_decompose(A);
        }
        ~lup_decomp_s()
        {}

        void solve(const matrix_type& A, const vector_type& b, vector_type& x)
        {

            if(wrong_size_) throw(std::runtime_error("lup_solve: incorrect matrix size provided: rows = " + std::to_string(sz_rows_) + " cols = " +  std::to_string(sz_cols_) ) );
            auto N = sz_rows_;
            for(size_t i = 0; i < N; i++)
            {
                x[i] = b[P_[i]];
                for (size_t k = 0; k < i; k++)
                {
                    x[i] -= A[I2_R(i,k,N)] * x[k];
                }
            }

            
            for (size_t i = N;  i--;)
            {
                for (size_t k = i + 1; k < N; k++)
                {
                    x[i] -= A[I2_R(i,k,N)] * x[k];
                }
                x[i] /= A[I2_R(i,i,N)];
            }
        }        
        
        void inv(const matrix_type& A, matrix_type& inv_A)
        {
          
            if(wrong_size_) throw(std::runtime_error("lup_solve: incorrect matrix size provided: rows = " + std::to_string(sz_rows_) + " cols = " +  std::to_string(sz_cols_) ) );
            auto N = sz_rows_;            
            #pragma omp parallel for
            for(size_t j = 0; j < N; j++) 
            {
                for(size_t i = 0; i < N; i++)
                {
                    inv_A[I2_R(i,j,N)] = (P_[i] == j ? static_cast<scalar_type>(1.0) : static_cast<scalar_type>(0.0) );

                    for (size_t k = 0; k < i; k++)
                    {
                        inv_A[I2_R(i,j,N)] -= A[I2_R(i,k,N)] * inv_A[I2_R(k,j,N)];
                    }
                }

                for(size_t i = N; i--;)
                {
                    for (size_t k = i + 1; k < N; k++)
                    {
                        inv_A[I2_R(i,j,N)] -= A[I2_R(i,k,N)] * inv_A[I2_R(k,j,N)];
                    }
                    inv_A[I2_R(i,j,N)] /= A[I2_R(i,i,N)];
                }
            }
        }

        T det(const matrix_type& A) 
        {
            if(wrong_size_) throw(std::runtime_error("lup_solve: incorrect matrix size provided: rows = " + std::to_string(sz_rows_) + " cols = " +  std::to_string(sz_cols_) ) );
            auto N = sz_rows_;
            T det = A[I2_R(0,0,N)];

            for(size_t i = 1; i < N; i++)
            {
                det *= A[I2_R(i,i,N)];
            }

            return (P_[N] - N) % 2 == 0 ? det : -det;
        }

        void permutation_matrix(matrix_type& P)
        {
            auto N = sz_rows_;

            for(size_t k = 0;k<N;k++)
            {
                for(size_t j = 0;j<N;j++)
                {
                    if(P_[k] == j)
                    {
                        P[I2_R(j,k,N)] = 1;
                    }
                    else
                    {
                        P[I2_R(j,k,N)] = 0;
                    }
                }
            }

        }

    private:
        size_t sz_rows_, sz_cols_;
        bool wrong_size_;
        std::vector<size_t> P_;
        T machesp_;

        void lup_decompose(matrix_type& A) 
        {
            if(wrong_size_) throw(std::runtime_error("lup_decompose: incorrect matrix size provided: rows = " + std::to_string(sz_rows_) + " cols = " +  std::to_string(sz_cols_) ) );
            size_t N = sz_rows_;


            std::iota(P_.begin(), P_.end(), 0);
            

            for(size_t i = 0; i < N; i++)
            {
                T max_A = static_cast<scalar_type>(0.0);
                size_t imax = i;

                for(size_t k = i; k < N; k++)
                {
                    auto abs_A = abs( A[I2_R(k,i,N)] );
                    if(abs_A > max_A)
                    { 
                        max_A = abs_A;
                        imax = k;
                    }
                }

                if(max_A < machesp_)
                {
                    throw(std::runtime_error("lup_decompose: matrix is singular to working precision."));
                }

                if(imax != i)
                {
                    //pivoting P
                    auto j = P_[i];
                    P_[i] = P_[imax];
                    P_[imax] = j;
                    //pivoting rows of A                    
                    for(size_t l=0;l<N;l++)
                    {
                        auto val = A[I2_R(i,l,N)];
                        A[I2_R(i,l,N)] = A[I2_R(imax,l,N)];
                        A[I2_R(imax,l,N)] = val;
                    }
                    
                    //counting pivots starting from N (for determinant)
                    P_[N]++;
                }

                #pragma omp parallel for
                for(size_t j = i + 1; j < N; j++)
                {
                    A[I2_R(j,i,N)] /= A[I2_R(i,i,N)];
                    for (size_t k = i + 1; k < N; k++)
                    {
                        A[I2_R(j,k,N)] -= A[I2_R(j,i,N)] * A[I2_R(i,k,N)];
                    }
                }
            }

        }



    };






    T two_prod(T &t, T a, T b) const // [1], pdf: 71, 169, 198, 
    {
        T p = a*b;
        t = fma(a, b, -p);
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
    mutable matrix_type helper_matrix_;
    const VectorOperations* vec_ops_;

};

#endif