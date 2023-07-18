#ifndef __PERIODIC_ORBIT_SYSTEM_OPERATOR_SINGLE_SECTION_H__
#define __PERIODIC_ORBIT_SYSTEM_OPERATOR_SINGLE_SECTION_H__

#include <vector>
#include <time_stepper/detail/all_methods_enum.h>
#include <periodic_orbit/hyperplane.h>
#include <periodic_orbit/poincare_map_operator.h>
#include <periodic_orbit/glued_poincare_map_linear_operator.h>


namespace nonlinear_operators
{
namespace newton_method
{

template<class VectorOperations, class NonlinearOperator, class LinearOperator, class LinearSolver>
class system_operator_single_section
{
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;

public:
    system_operator_single_section(VectorOperations* vec_ops_p, LinearOperator* lin_op_p, LinearSolver* lin_solver_p):
    vec_ops_(vec_ops_p),
    lin_op_(lin_op_p),
    lin_solver_(lin_solver_p)
    {
        vec_ops_->init_vector(b_); vec_ops_->start_use_vector(b_);
    }
    ~system_operator_single_section()
    {
        vec_ops_->stop_use_vector(b_); vec_ops_->free_vector(b_);
    }

    
    template<class VecOfVecs>
    void set_hyperplanes_from_initial_guesses(NonlinearOperator* nonlin_op_p, const VecOfVecs& init_vecs, const std::vector<T>& lambdas_p)
    {
        nonlin_op_p->set_hyperplanes_from_initial_guesses(init_vecs, lambdas_p);
    }
    
    void set_hyperplane_from_initial_guesses(NonlinearOperator* nonlin_op_p, const T_vec& init_vec, const T lambda_p)
    {
        nonlin_op_p->set_hyperplane_from_initial_guesses(init_vec, lambda_p);
    }

    
    bool solve(NonlinearOperator* nonlin_op, const T_vec& x, const T lambda, T_vec& d_x)const
    {
        bool flag_lin_solver = false;
        nonlin_op->F(x, lambda, b_);
        flag_lin_solver = lin_solver_->solve(*lin_op_, b_, d_x);
        nonlin_op->reproject(d_x);
        return flag_lin_solver;
    }
private:
    VectorOperations* vec_ops_;
    NonlinearOperator* nonlin_op_;
    LinearOperator* lin_op_;
    LinearSolver* lin_solver_;
    mutable T_vec b_;

};

}
}




#endif // __PERIODIC_ORBIT_SYSTEM_OPERATOR_SINGLE_SECTION_H__