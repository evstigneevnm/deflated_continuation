#ifndef __ARNOLDI_PROCESS_H__
#define __ARNOLDI_PROCESS_H__

/**
*
*    General Arnoldi process that calls a system operator to form the Krylov basis matrix and Hessenberg matrix.
*
*/

#include <stdexcept>
#include <cmath>

namespace numerical_algos
{
namespace eigen_solvers 
{

template<class VectorOperations, class MatrixOperations, class SystemOperator, class Log>
class arnoldi_process
{
public:
    typedef typename VectorOperations::scalar_type T;
    typedef typename VectorOperations::vector_type T_vec;
    typedef typename MatrixOperations::matrix_type T_mat;


    arnoldi_process(VectorOperations* vec_ops_large_, VectorOperations* vec_ops_small_, MatrixOperations* mat_ops_large_, MatrixOperations* mat_ops_small_, SystemOperator* sys_op_, Log* log_, T orthogonalization_tolerance_ = T(1.0e-12)):
    sys_op(sys_op_),
    vec_ops_large(vec_ops_large_),
    vec_ops_small(vec_ops_small_),
    mat_ops_large(mat_ops_large_),
    mat_ops_small(mat_ops_small_),
    log(log_)
    {
        start_arrays();
        N = mat_ops_large->get_rows();
        m = mat_ops_large->get_cols();
        size_t L2_size = vec_ops_large->get_l2_size();
        orthogonalization_tolerance = orthogonalization_tolerance_*T(L2_size);
    }
    ~arnoldi_process()
    {
        stop_arrays();
    }

    //sutable for both single execution and for the IRA execution
    // for the IRA: 0<=k<m and v_in is set
    // both V_mat and H_mat are on device
    // returns residual at position H(m, m+1)
    T execute_arnoldi(size_t& k, T_mat V_mat, T_mat H_mat, T_vec v_in = nullptr)
    {
        if(k > m)
        {
            throw std::runtime_error("numerical_algos::eigen_solvers::arnoldi_process: execute_arnoldi k > m");
        }
        if(v_in == nullptr)
        {
            v_in = v_v;
            vec_ops_large->assign_random(v_in);
        }
        if(k==0) //constructing the first vector in the V_mat
        {
            T beta = vec_ops_large->normalize(v_in);
            if (std::isnan(beta) ) throw std::runtime_error("numerical_algos::eigen_solvers::arnoldi_process: initial vector norm is NAN.");
            if (beta < T(1.0e-12)) throw std::runtime_error("numerical_algos::eigen_solvers::arnoldi_process: initial vector norm is too small.");
            
            bool res_flag = sys_op->solve(v_in, v_out);

            T alpha = vec_ops_large->scalar_prod(v_in, v_out);
            mat_ops_large->set_matrix_column(V_mat, v_in, 0); //sets the first vector in the Krylov subspace
            vec_ops_large->swap(v_out, v_in); //v_out <-> v_in
            vec_ops_large->add_mul(-alpha, v_out, v_in);
            //reorthogonalization
            T cc = T(1.0); int iter = 0;
            while((cc>orthogonalization_tolerance)&&(iter<10))
            {
                iter++;
                cc = vec_ops_large->scalar_prod(v_in, v_out);
                vec_ops_large->add_mul(-cc, v_out, v_in);
                alpha+=cc;
            }
            if(iter == 10)
            {
                log->warning_f("numerical_algos::eigen_solvers::arnoldi_process: reorthogonalization process failed with tolerance %le after %i iterations.", cc, iter);
            }            
            //reorthogonalization ends
            mat_ops_small->set_matrix_value(H_mat, alpha, 0, 0);
        }
        for(int j=k+1;j<m;j++) //assemble the whole Krylov basis and Hessenberg matrix
        { 
            T beta = vec_ops_large->normalize(v_in);
            T check_norm = vec_ops_large->norm(v_in);

            mat_ops_small->set_matrix_value(H_mat, beta, j, j-1);
            mat_ops_large->set_matrix_column(V_mat, v_in, j);   //sets the j-th vector in the Krylov subspace

            bool res_flag = sys_op->solve(v_in, v_out);
            
            vec_ops_small->assign_scalar(T(0.0), h_v);
            vec_ops_small->assign_scalar(T(0.0), cc_v);

            mat_ops_large->mat2column_dot_vec(V_mat, j+1, 1.0, v_out, 0.0, h_v); //j+1 ?
            mat_ops_large->mat2column_mult_vec(V_mat, j+1, T(-1.0), h_v, T(1.0), v_out);
            //reorthogonalization
            T cc = T(1.0); int iter = 0;
            while((cc>orthogonalization_tolerance)&&(iter<10))
            {
                iter++;
                mat_ops_large->mat2column_dot_vec(V_mat, j+1, 1.0, v_out, 0.0, cc_v); //j+1 ?
                mat_ops_large->mat2column_mult_vec(V_mat, j+1, T(-1.0), cc_v, T(1.0), v_out);
                vec_ops_small->add_mul(T(1.0), cc_v, h_v);
                cc = vec_ops_small->norm(cc_v);
            }
            if(iter == 10)
            {
                log->warning_f("numerical_algos::eigen_solvers::arnoldi_process: reorthogonalization process failed with tolerance %le after %i iterations.", cc, iter);
            }
            //reorthogonalization ends
            vec_ops_large->assign(v_out, v_in); // v_out -> v_in
            mat_ops_small->set_matrix_column(H_mat, h_v, j);
        }
        T beta_residual = vec_ops_large->norm(v_in);

        return beta_residual;
    }

    T execute_arnoldi_schur(size_t& k, T_mat V_mat, T_mat H_mat, T_vec v_in)
    {
        T beta = vec_ops_large->normalize(v_in);
        if (std::isnan(beta) ) throw std::runtime_error("numerical_algos::eigen_solvers::arnoldi_process: initial vector norm is NAN.");
        if (beta < 1.0e-12) throw std::runtime_error("numerical_algos::eigen_solvers::arnoldi_process: initial vector norm is too small.");        
        // size_t add = k>0?0:0;
        T ritz_estimate = 1.0;

        for(size_t j = k;j<m;j++)
        {
            // std::stringstream ssv;
            // ssv << "v_in_" << j << ".dat";
            // write_vector(ssv.str(), N, vec_ops_large->view(v_in));
            // auto x1 = vec_ops_large->view(v_in);
            mat_ops_large->set_matrix_column(V_mat, v_in, j);  //V_mat[k+1]<-v_in
            bool res_flag = sys_op->solve(v_in, v_out);
            // auto x2 = vec_ops_large->view(v_out);
            vec_ops_small->assign_scalar(T(0.0), h_v);
            vec_ops_small->assign_scalar(T(0.0), cc_v);
            mat_ops_large->mat2column_dot_vec(V_mat, j+1, 1.0, v_out, 0.0, h_v); //j+1 ?
            mat_ops_large->mat2column_mult_vec(V_mat, j+1, T(-1.0), h_v, T(1.0), v_out);                        
            T cc = 1.0; 
            int iter = 0;
            while((cc>orthogonalization_tolerance)&&(iter<10))
            {
                iter++;
                mat_ops_large->mat2column_dot_vec(V_mat, j+1, 1.0, v_out, 0.0, cc_v); //j+1 ?
                mat_ops_large->mat2column_mult_vec(V_mat, j+1, T(-1.0), cc_v, T(1.0), v_out);
                vec_ops_small->add_mul(T(1.0), cc_v, h_v);
                cc = vec_ops_small->norm(cc_v);
            }       

            // std::stringstream ss;
            // ss << "h_v_" << j << ".dat";
            // write_vector(ss.str(), m, vec_ops_small->view(h_v));
            // write_vector("v_out.dat", m, vec_ops_large->view(v_out));
            vec_ops_large->assign(v_out, v_in); 
            mat_ops_small->set_matrix_column(H_mat, h_v, j);
            ritz_estimate = vec_ops_large->normalize(v_in);
            if( std::isnan(ritz_estimate) )
            {
                throw std::runtime_error("numerical_algos::eigen_solvers::arnoldi_process: normalization of vector at step " + std::to_string(j) + " caused NAN." );
            }
            // std::cout << "Ritz_est = " << ritz_estimate << std::endl;
            if(j+1<m)
            {
                mat_ops_small->set_matrix_value(H_mat, ritz_estimate, j+1, j);
            }
        }


        return ritz_estimate;
    }



private:
    T_vec v_v = nullptr;
    T_vec v_out = nullptr;
    T_vec h_v = nullptr;
    T_vec cc_v = nullptr;
    size_t N;
    size_t m;
    T orthogonalization_tolerance;

    SystemOperator* sys_op;
    VectorOperations* vec_ops_large;
    VectorOperations* vec_ops_small;
    MatrixOperations* mat_ops_large;
    MatrixOperations* mat_ops_small;
    Log* log;


    void start_arrays()
    {
        vec_ops_small->init_vector(h_v); vec_ops_small->start_use_vector(h_v); 
        vec_ops_small->init_vector(cc_v); vec_ops_small->start_use_vector(cc_v); 
        vec_ops_large->init_vector(v_out); vec_ops_large->start_use_vector(v_out); 
        vec_ops_large->init_vector(v_v); vec_ops_large->start_use_vector(v_v); 
    }
    void stop_arrays()
    {
      
        vec_ops_small->stop_use_vector(h_v); vec_ops_small->free_vector(h_v); 
        vec_ops_small->stop_use_vector(cc_v); vec_ops_small->free_vector(cc_v); 
        vec_ops_large->stop_use_vector(v_out); vec_ops_large->free_vector(v_out); 
        vec_ops_large->stop_use_vector(v_v); vec_ops_large->free_vector(v_v); 

    }


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

};


}
}


#endif // __ARNOLDI_PROCESS_H__