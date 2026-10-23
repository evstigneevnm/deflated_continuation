#ifndef __DEFLATION_HPP__
#define __DEFLATION_HPP__


/**
*	 main mart of the deflation-continuation process.
*
*    Deflation class that utilizes single step deflation to find a solution.
*    If 'find_solution' returns false, then no more solutions are avaiable for the provided parameter value
*/

#include <deflation/system_operator_deflation.h>
#include <deflation/convergence_strategy.h>
#include <deflation/deflation_operator.h>
#include <memory>
#include <type_traits>
#include <numerical_algos/newton_solvers/newton_solver_extended.h>


namespace deflation
{

template <
    class VectorOperations, class VectorFileOperations, class Log, class NonlinearOperations, class LinearOperator,
    class LinearSolver, class SolutionStorage, class NewtonTypes = void>
class deflation
{
private:
    typedef typename VectorOperations::scalar_type T;
    typedef typename VectorOperations::vector_type T_vec;


    typedef SolutionStorage sol_storage_def_t;

    struct legacy_types
    {
        using deflation_convergence_type =
            newton_method_extended::convergence_strategy<VectorOperations, NonlinearOperations, Log>;
        using deflation_correction_type = numerical_algos::newton_method_extended::newton_solver_extended<
            VectorOperations, NonlinearOperations,
            system_operator_deflation<
                VectorOperations, NonlinearOperations, LinearOperator, LinearSolver, SolutionStorage, Log>,
            deflation_convergence_type, T>;
    };
    using selected_types           = std::conditional_t<std::is_void_v<NewtonTypes>, legacy_types, NewtonTypes>;
    using convergence_newton_def_t = typename selected_types::deflation_convergence_type;

    typedef system_operator_deflation<
        VectorOperations, NonlinearOperations, LinearOperator, LinearSolver, sol_storage_def_t, Log>
        system_operator_def_t;

    using newton_def_t = typename selected_types::deflation_correction_type;


    typedef deflation_operator<
        VectorOperations, newton_def_t, NonlinearOperations, sol_storage_def_t, Log, std::is_void_v<NewtonTypes>>
        deflation_operator_t;

public:
    deflation(
        VectorOperations *&vec_ops_, VectorFileOperations *&file_ops_, Log *&log_, NonlinearOperations *&nonlin_op_,
        LinearOperator *&lin_op_, LinearSolver *&SM_, SolutionStorage *&solution_storage_,
        std::shared_ptr<newton_def_t> correction = nullptr
    )
        : vec_ops( vec_ops_ ), file_ops( file_ops_ ), log( log_ ), nonlin_op( nonlin_op_ ),
          solution_storage( solution_storage_ )
    {
        if constexpr ( std::is_void_v<NewtonTypes> )
        {
            conv_newton_def     = new convergence_newton_def_t( vec_ops, log );
            system_operator_def = new system_operator_def_t( vec_ops, lin_op_, SM_, solution_storage, log );
            newton_def          = new newton_def_t( vec_ops, system_operator_def, conv_newton_def );
        }
        else
        {
            if ( !correction )
                throw std::invalid_argument( "Deflation requires an assembled correction" );
            correction_owner_ = std::move( correction );
            newton_def        = correction_owner_.get();
            conv_newton_def   = newton_def->get_convergence_strategy_handle();
        }
        deflation_op = new deflation_operator_t( vec_ops, log, newton_def );
    }
    ~deflation()
    {
        delete deflation_op;
        if constexpr ( std::is_void_v<NewtonTypes> )
        {
            delete newton_def;
            delete system_operator_def;
            delete conv_newton_def;
        }
    }

    void set_newton(
        T tolerance_, unsigned int maximum_iterations_, T newton_wight_ = T( 1 ), bool store_norms_history_ = false,
        bool verbose_ = true
    )
    {
        conv_newton_def->set_convergence_constants(
            tolerance_, maximum_iterations_, newton_wight_, store_norms_history_, verbose_
        );
    }
    void set_max_retries( unsigned int max_retries_ )
    {
        deflation_op->set_max_retries( max_retries_ );
    }

    void set_seed_sequence( const std::uint64_t first_seed )
    {
        deflation_op->set_seed_sequence( first_seed );
    }

    void clear_seed_sequence()
    {
        deflation_op->clear_seed_sequence();
    }

    std::uint64_t attempts_consumed() const
    {
        return deflation_op->attempts_consumed();
    }


    bool find_solution( const T &lambda_l )
    {
        bool result = deflation_op->find_solution( lambda_l, nonlin_op );
        return result;
    }
    void get_solution_ref( T_vec &sol_ref_ )
    {
        deflation_op->get_solution_ref( sol_ref_ );
    }


private:
    //passed:
    VectorOperations     *vec_ops;
    VectorFileOperations *file_ops;
    Log                  *log;
    NonlinearOperations  *nonlin_op;
    SolutionStorage      *solution_storage;

    //created localy:
    convergence_newton_def_t     *conv_newton_def;
    system_operator_def_t        *system_operator_def;
    newton_def_t                 *newton_def;
    deflation_operator_t         *deflation_op;
    std::shared_ptr<newton_def_t> correction_owner_;
};


}
#endif // __DEFLATION_HPP__
