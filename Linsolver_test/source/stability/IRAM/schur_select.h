#ifndef __STABILITY_IRAM_SCHUR_SELECT_H__
#define __STABILITY_IRAM_SCHUR_SELECT_H__

#include <limits>
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
class schur_select
{
private:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;
    typedef typename MatrixOperations::matrix_type T_mat;
    using C = std::complex<T>;
    using container_t = iram_container<VectorOperations, MatrixOperations, Log>;

    using eig_idx_sort_t = std::pair<std::complex<T>, size_t>;

    using vec_sort_t = std::pair< std::complex<T>, T>;

public:
    schur_select(VectorOperations* vec_ops_l_, MatrixOperations* mat_ops_l_, VectorOperations* vec_ops_s_, MatrixOperations* mat_ops_s_, Log* log_, LapackOperations* lapack_):
    vec_ops_l(vec_ops_l_),
    vec_ops_s(vec_ops_s_),
    mat_ops_l(mat_ops_l_),
    mat_ops_s(mat_ops_s_),
    log(log_),
    lapack(lapack_),
    block_ordering(true)
    {
        small_rows = mat_ops_s->get_rows();
        small_cols = mat_ops_s->get_cols();
        large_rows = mat_ops_l->get_rows();
        large_cols = mat_ops_l->get_cols();
        if(small_rows != small_cols)
        {
            throw std::logic_error("IRAM::schur_select: rows != cols of a small matrix class: rows = " + std::to_string(small_rows) + " cols = " + std::to_string(small_cols) );
        }
        if(large_cols != small_cols)
        {
            throw std::logic_error("IRAM::schur_select: number of cols for a large matrix class != rows of a small matrix class: small rows = " + std::to_string(small_rows) + " large cols = " + std::to_string(large_cols) );
        }    
        N = large_rows;
        m = small_cols;
        
        mu = std::vector<vec_sort_t>(m);
        
        Q = std::vector<T>(m*m,0); // Q'*Q = Q*Q' = E
        QC = std::vector<C>(m*m,0); // Q->to complex
        R = std::vector<T>(m*m,0); // block-quasi-triangular matrix (T in matlab)
        U = std::vector<C>(m*m,0); //eigenvectors
        U1 = std::vector<C>(m*m,0);
        eigs = std::vector<C>(m,0); 
        eigs1 = std::vector<C>(m,0); 
        ritz_c = std::vector<C>(m,0);
        ritz = std::vector<T>(m,0);
        H_host = std::vector<T>(m*m,0);
        hessinberg_mask = std::vector<T>(m*m,T(1.0) );
        H2_host = std::vector<T>(m*m,0);
        form_hessinberg_mask(); //do we need this mask???
        mat_ops_s->init_matrix(Q_gpu); mat_ops_s->start_use_matrix(Q_gpu);
        mat_ops_s->init_matrix(H_gpu); mat_ops_s->start_use_matrix(H_gpu);
        vec_ops_l->init_vector(f_gpu); vec_ops_l->start_use_vector(f_gpu);
        mat_ops_l->init_matrix(V_gpu); mat_ops_l->start_use_matrix(V_gpu);

    }
    ~schur_select()
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
    
    void set_number_of_desired_eigenvalues(int k_p)
    {
        if(k_p > m)
        {
            throw std::logic_error("IRAM::schur_select: number of desired eigenvalues = " + std::to_string(k_p) + " must not be greater then the maximum size of the Krylov subspace = " + std::to_string(m) );
        }
        k0 = k_p;
        k = k_p;
    }
    void set_target(std::string trg_)
    {
        target = trg_;
    }

    void set_block_ordering(bool ord_p)
    {
        block_ordering = ord_p;
    }


    void execute(container_t& cont_, T& ritz_value)
    {
        
        // std::cout << "ritz_value = " << ritz_value << std::endl;
        cont_.to_cpu();
        // lapack->hessinberg_schur(cont_.ref_H(), m, Q.data(), R.data(), (C*)eigs.data() ); // Q->T, R->U
        lapack->eigs_schur(cont_.ref_H(), m, eigs.data(), Q.data(), R.data());

        // write_matrix("Q_mat.dat", m, m, Q.data(), 4 );
        // write_matrix("R_mat.dat", m, m, R.data(), 4 );
        lapack->eigsv(R.data(), m, eigs.data(), U.data());
        // write_matrix("U_mat.dat", m, m, U.data(), 4 );
        // write_vector("eigR.dat", m, eigs.data(), 4 );
        lapack->double2complex(Q.data(), m, m, QC.data());
        lapack->gemm(QC.data(), 'N', U.data(), 'N', m, U1.data() );
        // write_matrix("QU_mat.dat", m, m, U1.data(), 4 );
        lapack->return_row(m-1, U1.data(), m, ritz_c.data() );
        // write_vector("ritz_vec.dat", m, ritz_c.data(), 4 );
        std::transform(ritz_c.begin(), ritz_c.end(), /*std::back_inserter(ritz_l)*/ ritz.begin(), [&ritz_value](const std::complex<T>& c1 ){ return ritz_value*std::abs<T>(c1); } );
        // write_vector("ritz_vec_abs.dat", m, ritz.data(), 4 );
        for(int j=0;j<m;j++)
        {
            mu.at(j).first = eigs.at(j);
            mu.at(j).second = ritz.at(j);
        }        
        sort_eigs(mu);
        for(int j=0;j<m;j++)
        {
            eigs.at(j) = mu.at(j).first;
            ritz.at(j) = mu.at(j).second;
            cont_.ritz.at(j) = mu.at(j).second;
        }
        // std::cout << "K = " << cont_.K << " k0 = " << k0 << " ritz_norm = " << cont_.ritz_norm() << std::endl;
        // write_vector("ritz_vec_abs_sorted.dat", m, ritz.data(), 4 );
        // write_vector("eigR_sorted.dat", m, eigs.data(), 4 );        
        if( cont_.ritz_norm() > cont_.get_tolerance() )
        {

            
            lapack->schur_upper_triag_ordered_eigs(R.data(), m, eigs1.data());
            // write_vector("eigR_ordered.dat", m, eigs1.data(), 4 );
            adjust_against_stagnation(cont_, ritz); //this updates K in container as well

            std::vector< eig_idx_sort_t > eigidx; //eigenvalues with indexes
            eigidx.reserve(m);
            for(int j=0;j<m;j++)
            {
                eigidx.push_back({eigs1.at(j), j});   
            }
            sort_eigs(eigidx);
            adjust_number_of_desired(cont_, eigidx);
            // for(auto& v: eigidx)
            // {
            //     std::cout << v.first << " " << v.second << " " << std::abs(v.first) << std::endl;
            // }
            
            std::vector<size_t> shifting_rows;
            shifting_rows.reserve(m);
            std::transform(eigidx.begin(), eigidx.end(), std::back_inserter(shifting_rows), [](const eig_idx_sort_t& pp){return pp.second;} );

            // lapack->reorder_schur(R.data(), Q.data(), m, 9, 2);
            // lapack->reorder_schur(R.data(), Q.data(), m, 11, 3);
            // lapack->reorder_schur(R.data(), Q.data(), m, 12, 5);
            // lapack->reorder_schur(R.data(), Q.data(), m, 14, 7);

            // ordered_schur(R, Q, shifting_rows);
            ordered_schur_block(R, Q, shifting_rows);

            // write_matrix("Q_mat_reord.dat", m, m, Q.data(), 4 );
            // write_matrix("R_mat_reord.dat", m, m, R.data(), 4 );
            //implicit restart
            std::vector<T> Q_row(m,0);
            lapack->return_row( m-1, Q.data(), m, Q_row.data() );
            std::transform(Q_row.begin(), Q_row.end(), Q_row.begin(), [&ritz_value](const T& r1 ){ return ritz_value*r1; } );
            // write_vector("Q_ritz_row.dat", m, Q_row.data(), 4 );
            std::vector<T> Q_sub(m*k,0);
            std::vector<T> R_sub(m*k,0);
            lapack->return_submatrix(Q.data(), {m,m}, {0,0}, Q_sub.data(), {m,k} );
            lapack->return_submatrix(R.data(), {m,m}, {0,0}, R_sub.data(), {m,k} );
            // write_matrix("Q_sub_mat.dat", m, k, Q_sub.data(), 4 );
            // write_matrix("R_sub_mat.dat", m, k, R_sub.data(), 4 );

            // write_matrix("H_before.dat", m, m, cont_.ref_H(), 4 );
            lapack->set_submatrix({0,0}, R_sub.data(), {m, k}, cont_.ref_H(), {m, m} );
            lapack->set_row(k, Q_row.data(), cont_.ref_H(), m, k);
            // write_matrix("H_after.dat", m, m, cont_.ref_H(), 4 );
            cont_.to_gpu();
            device_2_device_cpy(cont_.ref_V(), V_gpu, N*m);
            host_2_device_cpy(Q_gpu, Q.data(), m*m);

            mat_ops_l->mat2column_mult_mat(V_gpu, Q_gpu, m, 1.0, 0.0, cont_.ref_V() );
            // cont_.to_cpu();
            // write_matrix("V.dat", N, m, cont_.ref_V(), 4 );
            // write_vector("f_ref.dat", N, cont_.ref_f(), 4 );
            cont_.to_gpu();
        }
        else
        {
            
            int iter = 0;
            if(k0 < m)
            {
                if(eigs.at(k0-1) == conj(eigs.at(k0)) )
                {
                    ++k0;
                    ++cont_.K0;
                }
            }
            for(auto& eig: eigs)
            {
                ++iter;
                log->info_f("(%le, %le)", eig.real(), eig.imag() );
                if(iter>=k0)                
                {
                    break;
                }

            }
        }
        
    }

    //debug:
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
    template <class T>
    void write_vector(const std::string &f_name, size_t N, T *vec, unsigned int prec=17)
    {
        std::ofstream f(f_name.c_str(), std::ofstream::out);
        if (!f) throw std::runtime_error("print_matrix: error while opening file " + f_name);
        for (size_t i = 0; i<N; i++)
        {
            f << std::setprecision(prec) << vec[i] << std::endl;
        } 
        f.close();
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




private:
    bool block_ordering;
    int _deb_run_i_qr = 0;
    std::string target;
    std::vector<vec_sort_t> mu;
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
    size_t k = 6; //<-converged number of eigenvalues
    size_t k0 = 6; //<- wanted eigenvalues

    std::vector<T> Q;
    T_mat Q_gpu = nullptr;
    T_mat H_gpu = nullptr;
    T_vec f_gpu = nullptr;
    T_mat V_gpu = nullptr;
    std::vector<T> R;
    std::vector<C> eigs;
    std::vector<C> eigs1;
    std::vector<C> U;
    std::vector<C> U1;
    std::vector<C> QC;
    std::vector<T> ritz;
    std::vector<C> ritz_c;
    std::vector<T> H_host;
    std::vector<T> H2_host;
    size_t n_converged_old = 0;
    std::vector<T> hessinberg_mask;



    void ordered_schur(std::vector<T>& R, std::vector<T>& Q, std::vector<size_t>& shifting_rows)
    {

        std::vector<bool> subdiag(m);
        for(size_t l=0;l<m-1;l++)
        {
            subdiag.at(l) = std::abs(R[I2_R(l+1,l,m)])>std::numeric_limits<T>::epsilon();
        }        
        size_t j = 0;
        while(true)
        {
            auto shift_index = shifting_rows.at(j);
            lapack->reorder_schur(R.data(), Q.data(), m, shift_index+1, j+1);
            for(int l=0;l<m;l++)
            {
                if( shifting_rows.at(l) <= shift_index)
                {
                    ++shifting_rows.at(l);
                    if( subdiag.at(shift_index) )
                    {
                        ++shifting_rows.at(l);
                    }
                }
            }
            if( subdiag.at(shift_index) )
            {
                ++j;
                ++j;
            }
            else
            {
                ++j;
            }
            if(j>k)
            {
                break;
            }
        }        

    }
    void ordered_schur_block(std::vector<T>& R, std::vector<T>& Q, std::vector<size_t>& shifting_rows)
    {

        std::vector<bool> subdiag(m, false);
        std::vector<bool> select(m, false);
        for(size_t l=0;l<m-1;l++)
        {
            subdiag.at(l) = std::abs(R[I2_R(l+1,l,m)])>std::numeric_limits<T>::epsilon();           
        }        
        for(size_t l=0;l<k;l++)
        {
            select.at(shifting_rows.at(l)) = true;
        }


        size_t j = 0;
        for(size_t l=0;l<m;l++)
        {
            if(select.at(l))
            {   
                lapack->reorder_schur(R.data(), Q.data(), m, l+1, j+1);
                if( subdiag.at(l) )
                {
                    ++l;
                    ++j;
                    ++j;
                }
                else
                {
                    ++j;
                }
            }
 
        }        

    }

    void adjust_against_stagnation(container_t& cont_, const std::vector<T>& ritz)
    {
        size_t n_converged = 0;
        for(int j = 0; j<k0;j++)
        {
            n_converged += static_cast<size_t>( ritz.at(j)<=cont_.get_tolerance() );
        }
        
        if(n_converged < k0)
        {
            // Adjust k to prevent stagnating (Lehoucq, R.B. , D.C. Sorensen and C. Yang, ARPACK Users' Guide, SIAM, 1998)
            k = k0 + std::min(n_converged, static_cast<size_t>(std::floor( 0.5*(m - k0) ) ) );
            if((k==1)&&(m>3))
            {
                k = static_cast<size_t>(std::floor(0.5*m));
            }
            // Lola's heuristic
            if(((k + 1) < m) && (n_converged_old > n_converged))
            {
                k = k + 1;
            }
            n_converged_old = n_converged;
            cont_.K = k;
        }
    }

    template<class VV>
    void sort_eigs(std::vector< VV >& eigidx_p)
    {
        if((target == "LM")||(target == "lm"))
        {
            std::stable_sort(eigidx_p.begin(), eigidx_p.end(), [this](const VV& left_, const VV& right_)
            {
                return std::abs(left_.first) > std::abs(right_.first);
            } 
            );
        }
        else if((target == "LR")||(target == "lr"))
        {
            std::stable_sort(eigidx_p.begin(), eigidx_p.end(), [this](const VV& left_, const VV& right_)
            {
                return  left_.first.real() > right_.first.real();
            } 
            );
        }
        else if((target == "SR")||(target == "sr"))
        {
            std::stable_sort(eigidx_p.begin(), eigidx_p.end(), [this](const VV& left_, const VV& right_)
            {
                return  left_.first.real() < right_.first.real();
            } 
            );
        }        
    }


    // void adjust_number_of_desired(container_t& cont_, const std::vector< eig_idx_sort_t >& eigidx_p)
    // {
    //     size_t total_num = k;
    //     size_t j = 0;
    //     while(true)
    //     {
    //         auto eig1 = eigidx_p.at(j).first;
    //         auto eig2 = eigidx_p.at(j+1).first;
    //         if( eig1 == conj(eig2) )
    //         {
    //             ++total_num;
    //         }
    //         ++j;
    //         if(j >= total_num)
    //         {
    //             break;
    //         }
    //     }
    //     k = total_num;
    //     cont_.K = k;
    // }

    void adjust_number_of_desired(container_t& cont_, const std::vector< eig_idx_sort_t >& eigidx_p)
    {
        auto eig1 = eigidx_p.at(k-1).first;
        auto eig2 = eigidx_p.at(k).first;
        if(eig1 == conj(eig2))
        {
            ++k;
        }
        cont_.K = k;
    }
};

}
}


#endif