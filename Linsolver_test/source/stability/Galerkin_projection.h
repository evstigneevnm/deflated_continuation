#ifndef __STABILITY_GALERKIN_PROGECTION_H__
#define __STABILITY_GALERKIN_PROGECTION_H__ 

#include <complex>
#include <string>
#include <vector>
#include <utility>
#include <stdexcept>
#include <iomanip>
#include <algorithm>
#include <utils/cuda_support.h>
#include <stability/detail/eigenvalue_sorter.h>

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
    using eig_t = typename Eigenvalues::value_type;

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
        vec_ops_l->init_vector(w_helper); vec_ops_l->start_use_vector(w_helper);
        vec_ops_l->init_vector(v_r_); vec_ops_l->start_use_vector(v_r_);
        vec_ops_l->init_vector(v_i_); vec_ops_l->start_use_vector(v_i_);
        vec_ops_l->init_vector(w_r_); vec_ops_l->start_use_vector(w_r_);
        vec_ops_l->init_vector(w_i_); vec_ops_l->start_use_vector(w_i_);
        vec_ops_l->init_vector(eigv_real); vec_ops_l->start_use_vector(eigv_real);
        vec_ops_l->init_vector(eigv_imag); vec_ops_l->start_use_vector(eigv_imag);
        

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
        vec_ops_l->stop_use_vector(w_helper); vec_ops_l->free_vector(w_helper);
        vec_ops_l->stop_use_vector(v_r_); vec_ops_l->free_vector(v_r_);
        vec_ops_l->stop_use_vector(v_i_); vec_ops_l->free_vector(v_i_);
        vec_ops_l->stop_use_vector(w_r_); vec_ops_l->free_vector(w_r_);
        vec_ops_l->stop_use_vector(w_i_); vec_ops_l->free_vector(w_i_);        
        vec_ops_l->stop_use_vector(eigv_real); vec_ops_l->free_vector(eigv_real);
        vec_ops_l->stop_use_vector(eigv_imag); vec_ops_l->free_vector(eigv_imag);

    }
    void set_target_eigs(const std::pair<std::string, std::string>& which_)
    {
        sorter_system_operator.set_target_eigs(which_.first);
        sorter_orig.set_target_eigs(which_.second);
    }

/*
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
        sorter(eigs);


        return eigs;        

    }
*/

    eigs_t eigs(const T_mat& H, size_t k)
    {
        eigs_t eigs(m, 0);

        std::vector<T> Q(m*m, 0);
        std::vector<eig_t> QC(m*m, 0);   
        std::vector<T> R(m*m, 0);
        std::vector<eig_t> U(m*m, 0);
        std::vector<eig_t> U1(m*m, 0);

        lapack->eigs_schur_from_gpu(H, m, Q.data(), R.data(), eigs.data() );
        lapack->eigsv(R.data(), m, eigs.data(), U.data());
        sorter_orig(eigs);
        eigs.resize(k);
        return eigs;
    }



    eigs_t eigs(const T_mat& V, const T_mat& H, const size_t k)
    {
        std::vector<T_vec> fake_vec;
        return eigs(V, H, k, fake_vec, fake_vec);
    }
    eigs_t eigs(const T_mat& V, const T_mat& H, const size_t k, std::vector<T_vec>& eigvs_real, std::vector<T_vec>& eigvs_imag)    
    {
        

        eigs_t eigs(m, 0);
        std::vector< std::pair<eig_t, size_t> > eigsidxes(m);

        std::vector<T> Q(m*m, 0);
        std::vector<eig_t> QC(m*m, 0);   
        std::vector<T> R(m*m, 0);
        std::vector<eig_t> U(m*m, 0);
        std::vector<eig_t> U1(m*m, 0);

        lapack->eigs_schur_from_gpu(H, m, Q.data(), R.data(), eigs.data() );
        
        lapack->eigsv(R.data(), m, eigs.data(), U.data());
        size_t idx_e = 0;
        for(auto& e: eigs)
        {
            eigsidxes.at(idx_e) = {e, idx_e};
            idx_e++;
        }
        sorter_system_operator(eigsidxes);
        lapack->double2complex(Q.data(), m, m, QC.data());
        lapack->gemm(QC.data(), 'N', U.data(), 'N', m, U1.data() );
        
        std::vector<T> U_real;
        std::vector<T> U_imag;
        U_real.reserve(m*m);
        U_imag.reserve(m*m);

        idx_e = 0;
        for(auto& v: U1)
        {
            U_real.push_back( v.real() );
            U_imag.push_back( v.imag() );
        }
        host_2_device_cpy(Q_fin_dev, U_real.data(), m*m);
        mat_ops_l->mat2column_mult_mat(V, Q_fin_dev, m, 1.0, 0.0, V0); //V0: real eigenvectors
        host_2_device_cpy(Q_fin_dev, U_imag.data(), m*m);
        mat_ops_l->mat2column_mult_mat(V, Q_fin_dev, m, 1.0, 0.0, V1); //V1: imag eigenvectors
        eigs.resize(k);
        for(int j=0;j<k;j++)
        {
            auto sortidx = eigsidxes.at(j).second;
            vec_ops_l->assign( &V0[sortidx*N], eigv_real );
            vec_ops_l->assign( &V1[sortidx*N], eigv_imag );
            // std::stringstream ssr;
            // ssr << "eigv_real_" << j << ".dat";
            // write_vector(ssr.str(), N, vec_ops_l->view(eigv_real), 4);
            // std::stringstream ssi;
            // ssi << "eigv_imag_" << j << ".dat";
            // write_vector(ssi.str(), N, vec_ops_l->view(eigv_imag), 4);
 
            A->apply(eigv_real, &V0[j*N] ); //Avr->wr
            A->apply(eigv_imag, &V1[j*N] ); //Avi-->wi

            vec_ops_l->mul_pointwise(1.0, eigv_real, 1.0, eigv_real, v_r_); //vr^2
            vec_ops_l->mul_pointwise(1.0, eigv_imag, 1.0, eigv_imag, v_i_); //wr^2
            vec_ops_l->add_mul(1.0, v_r_, 1.0, v_i_, 0.0, v_helper); // vr^2+wr^2->v_helper

            vec_ops_l->mul_pointwise(1.0, eigv_real, 1.0, &V0[j*N], v_r_); // vr*wr
            vec_ops_l->mul_pointwise(1.0, eigv_imag, 1.0, &V1[j*N], v_i_); // vi*wi
            vec_ops_l->add_mul(1.0, v_r_, 1.0, v_i_, 0.0, w_helper); // vr*wr+vi*wi->v_helper
            vec_ops_l->div_pointwise(w_helper, 1.0, v_helper); //real parts of eigenvalues
            auto e_real = vec_ops_l->get_value_at_point(0, w_helper );

            vec_ops_l->mul_pointwise(1.0, eigv_real, 1.0, &V1[j*N], v_r_); // vr*wi
            vec_ops_l->mul_pointwise(1.0, eigv_imag, 1.0, &V0[j*N], v_i_); // vi*wi
            vec_ops_l->add_mul(1.0, v_r_, -1.0, v_i_, 0.0, w_helper); // vr*wi-vi*wr->v_helper
            vec_ops_l->div_pointwise(w_helper, 1.0, v_helper); //imag real parts of eigenvalues
            auto e_imag = vec_ops_l->get_value_at_point(0, w_helper );
            if((eigvs_real.size()>0)&&(eigvs_imag.size()>0))
            {
                vec_ops_l->assign( eigv_real, eigvs_real.at(j) );
                vec_ops_l->assign( eigv_imag, eigvs_imag.at(j) );
            }
            eigs.at(j) = {e_real, e_imag};
        }        


        return eigs;
    }



private:
    VectorOperations* vec_ops_l;
    MatrixOperations* mat_ops_l;
    VectorOperations* vec_ops_s;
    MatrixOperations* mat_ops_s;
    LapackOperations* lapack;
    LinearOperator* A;
    Log* log;  
    ::stability::detail::eigenvalue_sorter sorter_system_operator;
    ::stability::detail::eigenvalue_sorter sorter_orig;


    T_mat V0 = nullptr;
    T_mat V1 = nullptr;
    T_mat Q_fin_dev = nullptr;
    T_mat R_fin_dev = nullptr;
    T_mat H_dev = nullptr;
    T_vec v_helper = nullptr;
    T_vec w_helper = nullptr;
    T_vec v_r_ = nullptr;
    T_vec v_i_ = nullptr;
    T_vec w_r_ = nullptr;
    T_vec w_i_ = nullptr;
    T_vec eigv_real = nullptr;
    T_vec eigv_imag = nullptr;

    unsigned int m;
    size_t N;


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

};
}


#endif