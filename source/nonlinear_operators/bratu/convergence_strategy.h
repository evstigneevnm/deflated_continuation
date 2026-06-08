#ifndef __BRATU_CONVERGENCE_STRATEGY_H__
#define __BRATU_CONVERGENCE_STRATEGY_H__

#include <vector>

#include <common/scalar_math.h>

namespace nonlinear_operators
{
namespace newton_method
{

template<class VectorOperations, class NonlinearOperator, class Logging>
class convergence_strategy
{
public:
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;

    convergence_strategy(
        VectorOperations*& vec_ops_,
        Logging*& log_,
        T tolerance_ = T(1.0e-6),
        unsigned int maximum_iterations_ = 100,
        T newton_wight_ = T(1),
        bool store_norms_history_ = false,
        bool verbose_ = true):
        vec_ops(vec_ops_),
        log(log_),
        tolerance(tolerance_),
        maximum_iterations(maximum_iterations_),
        newton_wight(newton_wight_),
        newton_wight_initial(newton_wight_),
        verbose(verbose_),
        store_norms_history(store_norms_history_)
    {
        vec_ops->init_vector(x1);
        vec_ops->start_use_vector(x1);
        vec_ops->init_vector(Fx);
        vec_ops->start_use_vector(Fx);
    }

    ~convergence_strategy()
    {
        vec_ops->stop_use_vector(x1);
        vec_ops->free_vector(x1);
        vec_ops->stop_use_vector(Fx);
        vec_ops->free_vector(Fx);
    }

    void set_convergence_constants(
        T tolerance_,
        unsigned int maximum_iterations_,
        T newton_wight_ = T(1),
        bool store_norms_history_ = false,
        bool verbose_ = true)
    {
        tolerance = tolerance_;
        maximum_iterations = maximum_iterations_;
        newton_wight = newton_wight_;
        newton_wight_initial = newton_wight_;
        store_norms_history = store_norms_history_;
        verbose = verbose_;
        if(store_norms_history)
        {
            norms_evolution.reserve(maximum_iterations);
        }
    }

    bool check_convergence(
        NonlinearOperator* nonlin_op,
        T_vec& x,
        T lambda,
        T_vec& delta_x,
        int& result_status,
        bool lin_solver_converged = true)
    {
        if(!lin_solver_converged)
        {
            result_status = 5;
            return true;
        }

        nonlin_op->F(x, lambda, Fx);
        const T normFx = vec_ops->norm_l2(Fx);
        vec_ops->assign_mul(T(1), x, newton_wight, delta_x, x1);
        nonlin_op->F(x1, lambda, Fx);
        T normFx1 = vec_ops->norm_l2(Fx);
        if(store_norms_history)
        {
            norms_evolution.push_back(normFx1);
        }
        if(verbose)
        {
            log->info_f(
                "iteration %i, previous residual %le, current residual %le",
                iterations,
                static_cast<double>(normFx),
                static_cast<double>(normFx1));
        }

        if(normFx > T(0) && normFx1/normFx > T(2))
        {
            newton_wight *= T(0.75);
            vec_ops->assign_mul(T(1), x, newton_wight, delta_x, x1);
            nonlin_op->F(x1, lambda, Fx);
            normFx1 = vec_ops->norm_l2(Fx);
        }
        ++iterations;

        if(newton_wight < T(1.0e-6))
        {
            result_status = 4;
            return true;
        }
        if(common::scalar_math::isnan(normFx) || common::scalar_math::isnan(normFx1))
        {
            result_status = 3;
            return true;
        }
        if(common::scalar_math::isinf(normFx) || common::scalar_math::isinf(normFx1))
        {
            result_status = 2;
            return true;
        }

        vec_ops->assign(x1, x);
        if(normFx1 < tolerance)
        {
            result_status = 0;
            return true;
        }
        if(iterations >= maximum_iterations)
        {
            result_status = 1;
            return true;
        }
        return false;
    }

    unsigned int get_number_of_iterations()
    {
        return iterations;
    }

    void reset_iterations()
    {
        iterations = 0;
        newton_wight = newton_wight_initial;
        norms_evolution.clear();
    }

    void reset_wight()
    {
        newton_wight = newton_wight_initial;
    }

    std::vector<T>* get_norms_history_handle()
    {
        return &norms_evolution;
    }

private:
    VectorOperations* vec_ops;
    Logging* log;
    unsigned int iterations = 0;
    T tolerance;
    unsigned int maximum_iterations;
    T newton_wight;
    T newton_wight_initial;
    bool verbose;
    bool store_norms_history;
    T_vec x1;
    T_vec Fx;
    std::vector<T> norms_evolution;
};

} // namespace newton_method
} // namespace nonlinear_operators

#endif
