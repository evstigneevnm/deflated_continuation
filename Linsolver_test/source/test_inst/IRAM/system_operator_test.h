#ifndef __STABILITY__SYSTEM_OPERATOR_STABILITY_TEST_H__
#define __STABILITY__SYSTEM_OPERATOR_STABILITY_TEST_H__
    

/**
*
*  system operator for the tsting purposes
*
*/

#include <stdexcept>

namespace stability
{

template<class VectorOperations, class MatrixOperations, class Log>
class system_operator_stability
{
public:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;
    typedef typename MatrixOperations::matrix_type  T_mat;

    system_operator_stability(VectorOperations* vec_ops_, MatrixOperations* mat_ops_,  Log* log_):
    vec_ops(vec_ops_),
    mat_ops(mat_ops_),
    log(log_)
    {
        A_allocated = false;
    }

    ~system_operator_stability()
    {
        if(A_allocated)
        {
            mat_ops->stop_use_matrix(A); mat_ops->free_matrix(A);
        }
    }

    void copy_matrix(const T_mat matrix_)
    {
        mat_ops->init_matrix(A); mat_ops->start_use_matrix(A);
        A_allocated = true;
        mat_ops->assign(matrix_, A);

    }
    void set_matrix_ptr(const T_mat matrix_)
    {
        if(!A_allocated)
        {
            A = matrix_;
        }
        else
        {
            throw std::logic_error("attempt to set a marix pointer but a 'copy_matrix()' was already called and a matrix is assigned.");
        }
    }


    bool solve(const T_vec& v_in, T_vec& v_out)
    {
        
        mat_ops->gemv('N', A, 1.0, v_in, 0.0, v_out);
        return true;
    }

    std::string target_eigs()
    {
        return "LM";
    }


private:
    VectorOperations* vec_ops;
    MatrixOperations* mat_ops;
    Log* log;

    T_mat A;
    bool A_allocated;


};

}

#endif