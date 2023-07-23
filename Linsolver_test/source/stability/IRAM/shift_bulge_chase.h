#ifndef __STABILITY_IRAM_SHIFTS_BULGE_CHASE_H__
#define __STABILITY_IRAM_SHIFTS_BULGE_CHASE_H__

#include <vector>
#include <algorithm>
#include <complex>
#include <string>

//debug
#include <common/gpu_matrix_file_operations.h>
#include <utils/cuda_support.h>
#include <stability/IRAM/iram_container.h>


namespace stability
{
namespace IRAM
{

template<class VectorOperations, class MatrixOperations, class LapackOperations, class Log>
class shift_bulge_chase
{
private:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;
    typedef typename MatrixOperations::matrix_type T_mat;
    using C = std::complex<T>;
    using container_t = iram_container<VectorOperations, MatrixOperations, Log>;

    struct vec_sort
    {
        C lambda;
        T ritz;   
    };

public:
    shift_bulge_chase(VectorOperations* vec_ops_l_, MatrixOperations* mat_ops_l_, VectorOperations* vec_ops_s_, MatrixOperations* mat_ops_s_, Log* log_, LapackOperations* lapack_):
    vec_ops_l(vec_ops_l_),
    vec_ops_s(vec_ops_s_),
    mat_ops_l(mat_ops_l_),
    mat_ops_s(mat_ops_s_),
    log(log_),
    lapack(lapack_)
    {
        small_rows = mat_ops_s->get_rows();
        small_cols = mat_ops_s->get_cols();
        large_rows = mat_ops_l->get_rows();
        large_cols = mat_ops_l->get_cols();
        if(small_rows != small_cols)
        {
            throw std::logic_error("IRAM::shift_bulge_chase: rows != cols of a small matrix class: rows = " + std::to_string(small_rows) + " cols = " + std::to_string(small_cols) );
        }
        if(large_cols != small_cols)
        {
            throw std::logic_error("IRAM::shift_bulge_chase: number of cols for a large matrix class != rows of a small matrix class: small rows = " + std::to_string(small_rows) + " large cols = " + std::to_string(large_cols) );
        }    
        N = large_rows;
        m = small_cols;
        
        mu = std::vector<vec_sort>(m,{0,0});
    
        Q = std::vector<T>(m*m,0);
        R = std::vector<T>(m*m,0);
        eigs = std::vector<C>(m,0);
        ritz = std::vector<T>(m,0);
        H_host = std::vector<T>(m*m,0);
        hessinberg_mask = std::vector<T>(m*m,T(1.0) );
        H2_host = std::vector<T>(m*m,0);
        form_hessinberg_mask();
        mat_ops_s->init_matrix(Q_gpu); mat_ops_s->start_use_matrix(Q_gpu);
        mat_ops_s->init_matrix(H_gpu); mat_ops_s->start_use_matrix(H_gpu);
        vec_ops_l->init_vector(f_gpu); vec_ops_l->start_use_vector(f_gpu);
        mat_ops_l->init_matrix(V_gpu); mat_ops_l->start_use_matrix(V_gpu);

    }
    ~shift_bulge_chase()
    {
        if(Q_gpu != nullptr)
        {
            mat_ops_s->stop_use_matrix(Q_gpu); mat_ops_s->free_matrix(Q_gpu);
        }
        if(H_gpu != nullptr)
        {
            mat_ops_s->stop_use_matrix(H_gpu); mat_ops_s->free_matrix(H_gpu);
        }        
        if(f_gpu != nullptr)
        {
            vec_ops_l->stop_use_vector(f_gpu); vec_ops_l->free_vector(f_gpu);
        }
        if(V_gpu != nullptr)
        {
            mat_ops_l->stop_use_matrix(V_gpu); mat_ops_l->free_matrix(V_gpu);
        }
    }
    
    void set_number_of_desired_eigenvalues(unsigned int k_)
    {
        if(k_>m)
        {
            throw std::logic_error("IRAM::shift_bulge_chase: numner of desired eigenvalues = " + std::to_string(k_) + " must not be greater then the maximum size of the Krylov subspace = " + std::to_string(m) );
        }
        k = k_;

    }
    void set_target(std::string trg_)
    {
        target = trg_;
    }

    void select_shifts(const T_mat& H)
    {

        lapack->hessinberg_schur_from_gpu(H, m, Q.data(), R.data(), (C*)eigs.data() );
        lapack->return_row(m-1, Q.data(), m, ritz.data() );
        for(int j=0;j<m;j++)
        {
            mu.at(j).lambda = eigs.at(j);
            mu.at(j).ritz = std::abs(ritz.at(j));
        }
        sort_eigs();


    }

    void form_polynomial(const T_mat& H)
    {
        
        device_2_host_cpy(H_host.data(), H, m*m);

        lapack->eye(Q.data(), m);

        int j = m-1;
        while(j>=k)
        {
            auto lambda_ = mu.at(j).lambda;
            qr_step(lambda_);
            if(std::abs(lambda_.imag())>0.0)
            {
                j--;
                j--;
            }
            else
            {
                j--;
            }
        
        }
        k0 = j;
        
    }




    void transform_basis(container_t& cont_)
    {

        T f_norm = vec_ops_l->norm( cont_.ref_f() );
        cont_.K = k0;
        
        for(int j = 0;j<k;j++)
        {
            cont_.ritz.at(j) = f_norm*mu.at(j).ritz; //set scaled ritz estimates
        }
        // f = V*Q(:,ko+1)*H(ko+1,ko) + f*Q(m,ko);
        // V(:,1:ko) = V*Q(:,1:ko);
        T Q_val = Q.at((m-1)+m*(k0));
        T H_val = H_host.at((k0+1)+m*(k0));

        std::vector<T> Q_col(m,0);
        lapack->return_col( k0, Q.data(), m, Q_col.data() );
        host_2_device_cpy(Q_gpu, Q_col.data(), m);
        mat_ops_l->mat2column_mult_vec(cont_.ref_V(), m, H_val, Q_gpu, Q_val, cont_.ref_f() );
        host_2_device_cpy(Q_gpu, Q.data(), m*m);
        device_2_device_cpy(cont_.ref_V(), V_gpu, N*m);

        mat_ops_l->mat2column_mult_mat(V_gpu, Q_gpu, k0+1, 1.0, 0.0, cont_.ref_V() );

    }


    void execute(container_t& cont_)
    {
        select_shifts(cont_.ref_H() );
        form_polynomial(cont_.ref_H() );
        transform_basis(cont_);
        cont_.to_cpu();
        for(int j = 0;j<m*m;j++)
        {
            cont_.ref_H()[j] = H_host.at(j);
        }
        cont_.to_gpu();
    }

    //debug:
    void print_debug()
    {
        std::cout << "sorted lambdas:" << std::endl;
        for(int j=0;j<m;j++)
        {
            std::cout << mu.at(j).lambda;
            std::cout << std::endl;
        }  
        std::cout << "sorted Ritz-values:" << std::endl;
        for(int j=0;j<m;j++)
        {
            std::cout << mu.at(j).ritz;
            std::cout << std::endl;
        }          

    }
    void print_matrix(const T_mat& A)
    {

        for(int j =0;j<m;j++)
        {
            for(int l=0;l<m;l++)
            {
                std::cout << A[j+m*l] << " ";
            }
            std::cout << std::endl;
        }
    }  

    void _debug_set_Q(T_mat& Q_)
    {
        device_2_host_cpy(Q.data(), Q_, m*m);
    }
    void _debug_set_H(T_mat& H_)
    {
        device_2_host_cpy(H_host.data(), H_, m*m);
    }
    
    void _debug_print_eigs()
    {
        for(auto &x: mu)
        {
            log->info_f("%e, %e", x.lambda.real(), x.lambda.imag() );
        }
    }


private:
    // int I2_R(int i, int j, int Rows)
    // {
    //     return (i)+(j)*(Rows);
    // }
    template <class T>
    void write_matrix(const std::string &f_name, size_t Row, size_t Col, T *matrix, unsigned int prec=17)
    {
        size_t N=Col;
        std::ofstream f(f_name.c_str(), std::ofstream::out);
        if (!f) throw std::runtime_error("print_matrix: error while opening file " + f_name);
        for (size_t i = 0; i<Row; i++)
        {
            for(size_t j=0;j<Col;j++)
            {
                if(j<Col-1)
                    f << std::setprecision(prec) << matrix[I2_R(i,j,Row)] << " ";
                else
                    f << std::setprecision(prec) << matrix[I2_R(i,j,Row)];

            }
            f << std::endl;
        } 
        
        f.close();
    }

    void qr_step(C lambda_j) // submatrix is not used: int k1, int k2
    {
        T lambda_r = lambda_j.real();
        T lambda_i = lambda_j.imag();
        for(int l=0;l<m*m;l++)
            H2_host.at(l) = H_host.at(l);

        if(std::abs(lambda_i)>0.0)
        {
            lapack->add_to_diagonal(-lambda_r, m, H2_host.data());

            lapack->mat_sq(H2_host.data(), m, R.data());
            
            for(int l=0;l<m*m;l++)
                H2_host.at(l) = R.at(l);

            lapack->add_to_diagonal(lambda_i*lambda_i, m, H2_host.data());
        
        }
        else
        {
            lapack->add_to_diagonal(-lambda_r, m, H2_host.data() );
        }

        lapack->qr(H2_host.data(), m, R.data() ); //qr withough R, here R is actually Q

        lapack->gemm(R.data(), 'T', H_host.data(), 'N', m, H2_host.data() ); // H = Q'*H

        lapack->gemm(H2_host.data(), 'N', R.data(), 'N', m, H_host.data() ); // H = H*Q

        apply_hessinberg_mask(H_host.data() );
        
        for(int l=0;l<m*m;l++)
            H2_host.at(l) = Q.at(l);
        
        lapack->gemm(H2_host.data(), 'N', R.data(), 'N', m, Q.data() ); 
    }

    void apply_hessinberg_mask(T_mat A)
    {
        for(int j=0;j<m*m;j++)
        {
            A[j] *= hessinberg_mask[j];
        }
    }
    void form_hessinberg_mask()
    {
        for(int j = 0;j<m;j++)
        for(int l = 0;l<j-1;l++)
        {
            hessinberg_mask.at(j+l*m) = T(0.0); 
        }
    }

    void sort_eigs()
    {
        if((target == "LM")||(target == "lm"))
        {
            std::stable_sort(mu.begin(), mu.end(), [this](const vec_sort& left_, const vec_sort& right_)
            {
                return std::abs(left_.lambda) > std::abs(right_.lambda);
            } 
            );
        }
        else if((target == "LR")||(target == "lr"))
        {
            std::stable_sort(mu.begin(), mu.end(), [this](const vec_sort& left_, const vec_sort& right_)
            {
                return  left_.lambda.real() > right_.lambda.real();
            } 
            );
        }
    }




private:
    int _deb_run_i_qr = 0;
    std::string target;
    std::vector<vec_sort> mu;
    Log* log;
    VectorOperations* vec_ops_l;
    MatrixOperations* mat_ops_l;
    VectorOperations* vec_ops_s;
    MatrixOperations* mat_ops_s;
    LapackOperations* lapack;
    size_t small_rows;
    size_t small_cols;
    size_t large_rows;
    size_t large_cols;
    int m; //<-Krylov basis dim
    size_t N; //<-problem dim
    int k = 6; //<-wanted number of eigenvalues
    int k0; //<-converged wanted eigenvalues

    std::vector<T> Q;
    T_mat Q_gpu = nullptr;
    T_mat H_gpu = nullptr;
    T_vec f_gpu = nullptr;
    T_mat V_gpu = nullptr;
    std::vector<T> R;
    std::vector<C> eigs;
    std::vector<T> ritz;
    std::vector<T> H_host;
    std::vector<T> H2_host;

    std::vector<T> hessinberg_mask;

};

}
}

#endif