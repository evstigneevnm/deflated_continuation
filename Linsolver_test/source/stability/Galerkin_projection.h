#ifndef __STABILITY_GALERKIN_PROGECTION_H__
#define __STABILITY_GALERKIN_PROGECTION_H__ 

#include <string>
#include <vector>
#include <stdexcept>
#include <iomanip>
#include <algorithm>
#include <utils/cuda_support.h>

namespace stability
{

/**
 * @brief      Galerkin projection to obtain original eigenvalues. Can be used
 *             for arbitrary eigenvalue solvers and preconditioners.
 *
 * @tparam     VectorOperations  { N and m sized vector operations }
 * @tparam     MatrixOperations  { N-m and m-m sized matrix operations }
 * @tparam     LapackOperations  { Custom lapack operations (can be subsstituted
 *                               on both GPUs or CPUs) }
 * @tparam     LinearOperator    { Original linear operator }
 * @tparam     Log               { Log class }
 * @tparam     Eigenvalues       { Type that is used to store eigenvalues, compartable with std::vector over complex variables }
 */

template<class VectorOperations, class MatrixOperations, class LapackOperations, class LinearOperator, class Log, typename Eigenvalues>
class Galerkin_projection
{

private:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;
    typedef typename MatrixOperations::matrix_type T_mat;
    using eigs_t = Eigenvalues;
    using eig = typename Eigenvalues::value_type;

public:


    Galerkin_projection(VectorOperations* vec_ops_l_, MatrixOperations* mat_ops_l_, VectorOperations* vec_ops_s_, MatrixOperations* mat_ops_s_, LapackOperations* lapack_, LinearOperator* A_, Log* log_):
    vec_ops_l(vec_ops_l_), mat_ops_l(mat_ops_l_), vec_ops_s(vec_ops_s_), mat_ops_s(mat_ops_s_), lapack(lapack_), A(A_), log(log_)
    {

        mat_ops_l->init_matrix(V0); mat_ops_l->start_use_matrix(V0);
        mat_ops_l->init_matrix(V1); mat_ops_l->start_use_matrix(V1);
        mat_ops_s->init_matrix(H_dev); mat_ops_s->start_use_matrix(H_dev);
        mat_ops_s->init_matrix(Q_fin_dev); mat_ops_s->start_use_matrix(Q_fin_dev);
        mat_ops_s->init_matrix(R_fin_dev); mat_ops_s->start_use_matrix(R_fin_dev);

        vec_ops_l->init_vector(v_helper); vec_ops_l->start_use_vector(v_helper);

        auto small_rows = mat_ops_s->get_rows();
        auto small_cols = mat_ops_s->get_cols();
        auto large_rows = mat_ops_l->get_rows();
        auto large_cols = mat_ops_l->get_cols();
        if(small_cols != small_rows)
        {
            throw std::logic_error("Galerkin_projection: small matrix is not square");
        }
        if(small_rows != large_cols)
        {
            throw std::logic_error("Galerkin_projection: large_cols=" + std::to_string(large_cols) + " != small_rows=" + std::to_string(small_rows) );
        }
        N = large_rows;
        m = small_rows;

    }
    ~Galerkin_projection()
    {
        if(V0 != nullptr)
        {

            mat_ops_l->stop_use_matrix(V0); mat_ops_l->free_matrix(V0);
        }
        if(V1 != nullptr)
        {
            
            mat_ops_l->stop_use_matrix(V1); mat_ops_l->free_matrix(V1);
        }
        if( H_dev != nullptr)
        {
            
            mat_ops_s->stop_use_matrix(H_dev); mat_ops_s->free_matrix(H_dev);
        }
        if(Q_fin_dev != nullptr)
        {
            
            mat_ops_s->stop_use_matrix(Q_fin_dev); mat_ops_s->free_matrix(Q_fin_dev);
        }
        if(R_fin_dev != nullptr)
        {
            
            mat_ops_s->stop_use_matrix(R_fin_dev); mat_ops_s->free_matrix(R_fin_dev);
            
        }
        if(v_helper != nullptr)
        {
            vec_ops_l->stop_use_vector(v_helper); vec_ops_l->free_vector(v_helper);
        }


    }
    void set_target_eigs(const std::string& which_)
    {
        
        bool found = false;
        for(auto &p_: sorting_list_permisive)
        {
            if(which_ == p_)
            {
                found = true;
                break;
            }
        }
        if(!found)
        {
            throw std::logic_error("Galerkin_projection::set_target_eigs: Only LM, LR or SR sort of eigenvalues is supported. You provided: " + which);
        }
        which = which_;
    }
    eigs_t eigs(const T_mat& V, const T_mat& H, const size_t k)
    {
        eigs_t eigs = eigs_t(m, 0);
        mat_ops_l->make_zero_columns(V, k, m, V0 );
        mat_ops_s->make_zero_columns(H, k, m, H_dev);
        std::vector<T> Q_fin(m*m, 0);
        std::vector<T> U_fin(m*m, 0);
        lapack->eigs_schur_from_gpu(H_dev, m, Q_fin.data(), U_fin.data(), eigs.data() );
        eigs.resize(k);
        host_2_device_cpy(Q_fin_dev, Q_fin.data(), m*m);
        mat_ops_l->mat2column_mult_mat(V0, Q_fin_dev, m, 1.0, 0.0, V1);
        for(int j=0;j<k;j++)
        {
            
            A->apply(&V1[j*N], v_helper);
            vec_ops_l->assign(v_helper, &V0[j*N]);
            //&V0[j*N]);
        }
        mat_ops_l->mat_T_gemm_mat_N(V1, V0, m, 1.0, 0.0, R_fin_dev);
        std::vector<T> R_fin(m*m, 0);
        std::vector<T> R_fin_sub(k*k, 0);
        device_2_host_cpy(R_fin.data(), R_fin_dev, m*m);
        //copy a submatrix to a smaller matrix. Should be remade more conceptually.
        // for(int j=0;j<k;j++)
        // {
        //     for(int l=0;l<k;l++)
        //     {
        //         R_fin_sub[j+l*k] = R_fin[j+l*m];
        //     }
        // }
        lapack->return_submatrix(R_fin.data(), {m,m}, {0,0}, R_fin_sub.data(), {k,k});

        // eigs_t eigs = eigs_t(k, 0);
        lapack->eigs(R_fin_sub.data(), k, eigs.data() );
        sort_eigs(eigs);

        return eigs;        

    }

    // TODO: add implementation for eigenvectors
    // This requres more information from the sorting process i.e. the permuation index matrix


private:
    VectorOperations* vec_ops_l;
    MatrixOperations* mat_ops_l;
    VectorOperations* vec_ops_s;
    MatrixOperations* mat_ops_s;
    LapackOperations* lapack;
    LinearOperator* A;
    Log* log;  

    T_mat V0 = nullptr;
    T_mat V1 = nullptr;
    T_mat Q_fin_dev = nullptr;
    T_mat R_fin_dev = nullptr;
    T_mat H_dev = nullptr;
    T_vec v_helper = nullptr;

    unsigned int m;
    size_t N;
    std::string which = "LM"; // default
    std::vector<std::string> sorting_list_permisive = {"LR", "lr", "LM", "lm", "SR", "sr"};

    void sort_eigs(eigs_t& eigs)
    {
        if((which == "LM")||(which == "lm"))
        {
            std::stable_sort(eigs.begin(), eigs.end(), [this](const eig& left_, const eig& right_)
            {
             
                return std::abs(left_) > std::abs(right_);
            } 
            );
        }
        else if((which == "LR")||(which == "lr"))
        {
            std::stable_sort(eigs.begin(), eigs.end(), [this](const eig& left_, const eig& right_)
            {
                return  left_.real() > right_.real();
            } 
            );
        }
        else if((which == "SR")||(which == "sr"))
        {
            std::stable_sort(eigs.begin(), eigs.end(), [this](const eig& left_, const eig& right_)
            {
                return  left_.real() < right_.real();
            } 
            );
        }  
        else
        {
            throw std::logic_error("Galerkin_projection::sort: Only LM, LR or SR sort of eigenvalues is supported. You provided: " + which);
        }      
    }

};
}


#endif