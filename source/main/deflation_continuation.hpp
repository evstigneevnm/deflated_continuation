#ifndef __DEFLATION_CONTINUATION_HPP__
#define __DEFLATION_CONTINUATION_HPP__

/**
*   The main class of the whole Deflation-Continuaiton Process (DCP). 
*
*   It uses nonlinear operator and other set options to configure the whole project.
*   After vector and file operations, the nonlinear operator, log and monitor are configured,
*   this class is initialized and configured to perform the whole DCP.
*   data serialization is done using boost archive
*/
#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>
#include <limits>
#include <memory>
#include <sstream>
// #include <type_traits> //to check linsolvers
//boost serializatoin
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
// #include <boost/archive/binary_oarchive.hpp>
// #include <boost/archive/binary_iarchive.hpp>


#include <numerical_algos/lin_solvers/sherman_morrison_linear_system_solve.h>
#include <numerical_algos/newton_solvers/newton_solver.h>

#include <containers/knots.hpp>
#include <containers/knot_registry.h>
#include <containers/branch_intersection.h>
#include <containers/curve_helper_container.h>
#include <containers/bifurcation_diagram_curve.h>
#include <containers/bifurcation_diagram.h>

#include <continuation/continuation.hpp>
#include <continuation/continuation_analytical.hpp> // inherited from continuation to put a nontrivial analytical solution on the curve, if needed.

#include <deflation/solution_storage.h>
#include <deflation/deflation.hpp>



namespace main_classes{

namespace detail
{

template<class Solver>
auto set_use_precond_resid_if_available(Solver* solver, int value) -> decltype(solver->set_use_precond_resid(value), void())
{
    solver->set_use_precond_resid(value);
}

inline void set_use_precond_resid_if_available(...)
{
}

template<class Solver>
auto set_resid_recalc_freq_if_available(Solver* solver, int value) -> decltype(solver->set_resid_recalc_freq(value), void())
{
    solver->set_resid_recalc_freq(value);
}

inline void set_resid_recalc_freq_if_available(...)
{
}

template<class Solver>
auto set_basis_size_if_available(Solver* solver, int value) -> decltype(solver->set_basis_size(value), void())
{
    solver->set_basis_size(value);
}

inline void set_basis_size_if_available(...)
{
}

template<class SolutionStorage, class Vector>
auto stabilize_solution_if_available(SolutionStorage* storage, Vector& x) -> decltype(storage->stabilize_in_place(x), void())
{
    storage->stabilize_in_place(x);
}

inline void stabilize_solution_if_available(...)
{
}

template<class Continuation, class SolutionStorage, class Vector>
auto set_solution_postprocessor_if_available(Continuation* continuation, SolutionStorage* storage, Vector*) -> decltype(storage->stabilize_in_place(std::declval<Vector&>()), void())
{
    continuation->set_solution_postprocessor([storage](Vector& x)
    {
        storage->stabilize_in_place(x);
    });
}

inline void set_solution_postprocessor_if_available(...)
{
}

template<class Continuation>
auto set_allow_knot_interpolation_failure_if_available(Continuation* continuation, bool value) -> decltype(continuation->set_allow_knot_interpolation_failure(value), void())
{
    continuation->set_allow_knot_interpolation_failure(value);
}

inline void set_allow_knot_interpolation_failure_if_available(...)
{
}

template<class Continuation, class Resolver>
auto set_knot_resolver_if_available(Continuation* continuation, Resolver&& resolver)
    -> decltype(continuation->set_knot_resolver(std::forward<Resolver>(resolver)), void())
{
    continuation->set_knot_resolver(std::forward<Resolver>(resolver));
}

inline void set_knot_resolver_if_available(...)
{
}

template<class Continuation, class Relocator>
auto set_knot_relocator_if_available(Continuation* continuation, Relocator&& relocator)
    -> decltype(continuation->set_knot_relocator(std::forward<Relocator>(relocator)), void())
{
    continuation->set_knot_relocator(std::forward<Relocator>(relocator));
}

inline void set_knot_relocator_if_available(...)
{
}

template<class SolutionStorage, class Vector, class Scalar>
auto nearest_stabilized_distance_if_available(SolutionStorage* storage, Vector& x, Scalar& distance)
    -> decltype(storage->nearest_stabilized_distance(x), bool())
{
    distance = static_cast<Scalar>(storage->nearest_stabilized_distance(x));
    return true;
}

inline bool nearest_stabilized_distance_if_available(...)
{
    return false;
}

} // namespace detail


template<class VectorOperations, class VectorFileOperations, class Log, class Monitor, class NonlinearOperations, class LinearOperator, class Preconditioner, template<class , class , class , class , class > class LinearSolver, template<class , class , class , class > class SystemOperator, class Parameters, class SolutionStorage = deflation::solution_storage<VectorOperations, Log>, template<class, class, class, class, class> class ContinuationSystemOperator = continuation::system_operator_continuation>
class deflation_continuation
{
private:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;
    typedef Monitor monitor_t;

    typedef typename boost::archive::text_oarchive data_output;
    typedef typename boost::archive::text_iarchive data_input;

    //general linear solver used in continuation and deflation
    typedef numerical_algos::sherman_morrison_linear_system::sherman_morrison_linear_system_solve<
        LinearOperator,
        Preconditioner,
        VectorOperations,
        monitor_t,
        Log,
        LinearSolver
        > sherman_morrison_linear_system_solve_t;

    //general system operator for the newton's method
    //TODO: move to nonlinear_operators?
    typedef SystemOperator<
        VectorOperations, 
        NonlinearOperations,
        LinearOperator,
        sherman_morrison_linear_system_solve_t
        > system_operator_t;

    //convergence strategy for the newton's method for F(x) = 0
    typedef nonlinear_operators::newton_method::convergence_strategy<
        VectorOperations, 
        NonlinearOperations, 
        Log> convergence_newton_t;
    //newton's method for F(x) = 0
    typedef numerical_algos::newton_method::newton_solver<
        VectorOperations, 
        NonlinearOperations,
        system_operator_t, 
        convergence_newton_t
        > newton_t;
    
    typedef container::knots<T> knots_t;
    typedef container::knot_registry<T> knot_registry_t;

    typedef container::curve_helper_container<VectorOperations> container_helper_t;

    typedef SolutionStorage sol_storage_def_t;


    typedef container::bifurcation_diagram_curve<
        VectorOperations,
        VectorFileOperations, 
        Log,
        NonlinearOperations,
        newton_t, 
        sol_storage_def_t,
        container_helper_t
        > bif_diag_curve_t;

    typedef container::bifurcation_diagram<
        VectorOperations,
        VectorFileOperations, 
        Log,
        NonlinearOperations,
        newton_t, 
        sol_storage_def_t,
        bif_diag_curve_t,
        container_helper_t
        > bif_diag_t;

    typedef continuation::continuation<
        VectorOperations, 
        VectorFileOperations, 
        Log, 
        NonlinearOperations, 
        LinearOperator,  
        knots_t,
        sherman_morrison_linear_system_solve_t,  
        newton_t,
        bif_diag_curve_t,
        ContinuationSystemOperator
        > continuate_t;

    typedef continuation::continuation_analytical<
        VectorOperations, 
        VectorFileOperations, 
        Log, 
        NonlinearOperations, 
        LinearOperator,  
        knots_t,
        sherman_morrison_linear_system_solve_t,  
        newton_t,
        bif_diag_curve_t,
        ContinuationSystemOperator
        > continuate_analytical_t;

    typedef deflation::deflation<
        VectorOperations,
        VectorFileOperations,
        Log,
        NonlinearOperations,
        LinearOperator,
        sherman_morrison_linear_system_solve_t,
        sol_storage_def_t
        > deflate_t;

    typedef container::intersection_status intersection_status_t;

    class rejected_candidate_cache
    {
    public:
        explicit rejected_candidate_cache(VectorOperations* vec_ops_):
            vec_ops(vec_ops_)
        {
            vec_ops->init_vector(diff);
            vec_ops->start_use_vector(diff);
        }

        ~rejected_candidate_cache()
        {
            clear();
            vec_ops->stop_use_vector(diff);
            vec_ops->free_vector(diff);
        }

        void clear()
        {
            for(auto& x: rejected_vectors)
            {
                vec_ops->stop_use_vector(x);
                vec_ops->free_vector(x);
            }
            rejected_vectors.clear();
            lambdas.clear();
        }

        void add(const T& lambda, const T_vec& x)
        {
            T_vec x_copy;
            vec_ops->init_vector(x_copy);
            vec_ops->start_use_vector(x_copy);
            vec_ops->assign(x, x_copy);
            rejected_vectors.push_back(std::move(x_copy));
            lambdas.push_back(lambda);
        }

        bool nearest_distance(const T& lambda, const T_vec& x, T& distance)
        {
            bool found = false;
            distance = std::numeric_limits<T>::infinity();
            for(std::size_t i = 0; i < rejected_vectors.size(); ++i)
            {
                if(!same_lambda(lambda, lambdas[i]))
                {
                    continue;
                }
                vec_ops->assign_mul(T(1), x, T(-1), rejected_vectors[i], diff);
                const T current_distance = vec_ops->norm_l2(diff);
                if(current_distance < distance)
                {
                    distance = current_distance;
                    found = true;
                }
            }
            return found;
        }

    private:
        static bool same_lambda(const T& a, const T& b)
        {
            const T scale = std::max<T>(T(1), std::max<T>(scalar_abs(a), scalar_abs(b)));
            return scalar_abs(a - b) <= T(64)*std::numeric_limits<T>::epsilon()*scale;
        }

        static T scalar_abs(const T& value)
        {
            return value < T(0) ? -value : value;
        }

        VectorOperations* vec_ops;
        std::vector<T_vec> rejected_vectors;
        std::vector<T> lambdas;
        T_vec diff;
    };

    class branch_distance_workspace
    {
    public:
        branch_distance_workspace(VectorOperations* vec_ops_, sol_storage_def_t* storage_):
            vec_ops(vec_ops_),
            storage(storage_)
        {
            vec_ops->init_vector(a_work);
            vec_ops->start_use_vector(a_work);
            vec_ops->init_vector(b_work);
            vec_ops->start_use_vector(b_work);
            vec_ops->init_vector(diff);
            vec_ops->start_use_vector(diff);
        }

        ~branch_distance_workspace()
        {
            vec_ops->stop_use_vector(diff);
            vec_ops->free_vector(diff);
            vec_ops->stop_use_vector(b_work);
            vec_ops->free_vector(b_work);
            vec_ops->stop_use_vector(a_work);
            vec_ops->free_vector(a_work);
        }

        T distance(const T_vec& a, const T_vec& b)
        {
            vec_ops->assign(a, a_work);
            vec_ops->assign(b, b_work);
            detail::stabilize_solution_if_available(storage, a_work);
            detail::stabilize_solution_if_available(storage, b_work);
            vec_ops->assign_mul(T(1), a_work, T(-1), b_work, diff);
            return vec_ops->norm_l2(diff);
        }

    private:
        VectorOperations* vec_ops;
        sol_storage_def_t* storage;
        T_vec a_work;
        T_vec b_work;
        T_vec diff;
    };

    static bool same_parameter_value(const T& a, const T& b)
    {
        const T scale = std::max<T>(T(1), std::max<T>(scalar_abs(a), scalar_abs(b)));
        return scalar_abs(a - b) <= T(64)*std::numeric_limits<T>::epsilon()*scale;
    }

    static T scalar_abs(const T& value)
    {
        return value < T(0) ? -value : value;
    }


public:
    deflation_continuation(VectorOperations* vec_ops_, VectorFileOperations* file_ops_, Log* log_, Log* log_linsolver_, NonlinearOperations* nonlin_op_, Parameters* parameters_, sol_storage_def_t* sol_storage_external_ = nullptr):
    vec_ops(vec_ops_),
    file_ops(file_ops_),
    log(log_),
    nonlin_op(nonlin_op_),
    log_linsolver(log_linsolver_),
    parameters(parameters_),
    sol_storage_def(sol_storage_external_),
    owns_solution_storage(sol_storage_external_ == nullptr)
    {
        
        //add '/' to the end of the project dir, if needed
        
        project_dir = parameters->path_to_project;
        skip_files = parameters->deflation_continuation.skip_files;

        if(!project_dir.empty() && *project_dir.rbegin() != '/')
            project_dir += '/';


        lin_op = new LinearOperator(nonlin_op);
        precond = new Preconditioner(nonlin_op);
        SM = new sherman_morrison_linear_system_solve_t(precond, vec_ops, log_linsolver);
        conv_newton = new convergence_newton_t(vec_ops, log);
        system_operator = new system_operator_t(vec_ops, lin_op, SM);
        newton = new newton_t(vec_ops, system_operator, conv_newton);
        knots = new knots_t();
        continuate = new continuate_t(vec_ops, file_ops, log, nonlin_op, lin_op, knots, SM, newton);
        continuate_analytical = new continuate_analytical_t(vec_ops, file_ops, log, nonlin_op, lin_op, knots, SM, newton);
        bif_diag = new bif_diag_t(vec_ops, file_ops, log, nonlin_op, newton, project_dir, skip_files);
        if(sol_storage_def == nullptr)
        {
            sol_storage_def = new sol_storage_def_t(vec_ops, 50, vec_ops->get_l2_size(), 2.0, log );  //T(1.0) is a norm_wight! Used as sqrt(N) for L2 norm. Use it again? Check this!!!
        }
        detail::set_solution_postprocessor_if_available(continuate, sol_storage_def, static_cast<T_vec*>(nullptr));
        detail::set_solution_postprocessor_if_available(continuate_analytical, sol_storage_def, static_cast<T_vec*>(nullptr));
        deflate = new deflate_t(vec_ops, file_ops, log, nonlin_op, lin_op, SM, sol_storage_def);
    }
    ~deflation_continuation()
    {
        
        delete deflate;
        if(owns_solution_storage)
        {
            delete sol_storage_def;
        }
        delete bif_diag;
        delete continuate;
        delete continuate_analytical;
        delete knots;
        delete newton;
        delete system_operator;
        delete conv_newton;
        delete SM;
        delete precond;
        delete lin_op;
    }



//  called to set all parameters from the parameter structure
    void set_parameters()
    {
        set_linsolver();
        set_extended_linsolver();
        set_newton();
        set_newton_continuation();
        set_newton_deflation();
        set_steps();
        set_deflation_knots();
        set_branch_intersection_policy();
    }

    void set_linsolver()
/*T lin_solver_tol, unsigned int lin_solver_max_it, int use_precond_resid = 1, int resid_recalc_freq = 1, int basis_sz = 4, bool save_convergence_history_  = true, bool divide_out_norms_by_rel_base_ = true*/
    {
        //setup linear system:
        mon_orig = &SM->get_linsolver_handle_original()->monitor();
        T lin_solver_tol = parameters->nonlinear_operator.linear_solver.lin_solver_tol;
        unsigned int lin_solver_max_it = parameters->nonlinear_operator.linear_solver.lin_solver_max_it;
        bool save_convergence_history_ = parameters->nonlinear_operator.linear_solver.save_convergence_history;
        bool divide_out_norms_by_rel_base_ = parameters->nonlinear_operator.linear_solver.divide_out_norms_by_rel_base;
        bool verbose_ = parameters->nonlinear_operator.linear_solver.verbose;
        int use_precond_resid = parameters->nonlinear_operator.linear_solver.use_precond_resid;
        int resid_recalc_freq = parameters->nonlinear_operator.linear_solver.resid_recalc_freq;
        int basis_sz = parameters->nonlinear_operator.linear_solver.basis_size;

        mon_orig->init(lin_solver_tol, T(0), lin_solver_max_it);
        mon_orig->set_save_convergence_history(save_convergence_history_);
        mon_orig->set_divide_out_norms_by_rel_base(divide_out_norms_by_rel_base_);
        mon_orig->set_verbose(verbose_);
        mon_orig->out_min_resid_norm();
//
        if(use_precond_resid >= 0)
            detail::set_use_precond_resid_if_available(SM->get_linsolver_handle_original(), use_precond_resid);
        if(resid_recalc_freq >= 0)
            detail::set_resid_recalc_freq_if_available(SM->get_linsolver_handle_original(), resid_recalc_freq);
        if(basis_sz > 0)
            detail::set_basis_size_if_available(SM->get_linsolver_handle_original(), basis_sz);
            // SM->get_linsolver_handle_original()->set_restarts(basis_sz);  
//
    }
    void set_extended_linsolver()
/*T lin_solver_tol, unsigned int lin_solver_max_it, bool is_small_alpha = false, int use_precond_resid = 1, int resid_recalc_freq = 1, int basis_sz = 4, bool save_convergence_history_  = true, bool divide_out_norms_by_rel_base_ = true
*/    
    {
        mon = &SM->get_linsolver_handle()->monitor();
        T lin_solver_tol = parameters->deflation_continuation.linear_solver_extended.lin_solver_tol;
        unsigned int lin_solver_max_it = parameters->deflation_continuation.linear_solver_extended.lin_solver_max_it;
        bool save_convergence_history_ = parameters->deflation_continuation.linear_solver_extended.save_convergence_history;
        bool divide_out_norms_by_rel_base_ = parameters->deflation_continuation.linear_solver_extended.divide_out_norms_by_rel_base;
        bool verbose_ = parameters->deflation_continuation.linear_solver_extended.verbose;
        int use_precond_resid = parameters->deflation_continuation.linear_solver_extended.use_precond_resid;
        int resid_recalc_freq = parameters->deflation_continuation.linear_solver_extended.resid_recalc_freq;
        int basis_sz = parameters->deflation_continuation.linear_solver_extended.basis_size;
        bool is_small_alpha = parameters->deflation_continuation.linear_solver_extended.is_small_alpha;

        mon->init(lin_solver_tol, T(0), lin_solver_max_it);
        mon->set_save_convergence_history(save_convergence_history_);
        mon->set_divide_out_norms_by_rel_base(divide_out_norms_by_rel_base_);
        mon->set_verbose(verbose_);
        mon->out_min_resid_norm();
//
        if(use_precond_resid >= 0)
            detail::set_use_precond_resid_if_available(SM->get_linsolver_handle(), use_precond_resid);
        if(resid_recalc_freq >= 0)
            detail::set_resid_recalc_freq_if_available(SM->get_linsolver_handle(), resid_recalc_freq);
        if(basis_sz > 0)
           detail::set_basis_size_if_available(SM->get_linsolver_handle(), basis_sz);
            // SM->get_linsolver_handle()->set_restarts(basis_sz); 
//
        SM->is_small_alpha(is_small_alpha);        
    }
    void set_newton()
    /*T tolerance_, unsigned int maximum_iterations_, T newton_wight_ = T(0.5), bool store_norms_history_ = false, bool verbose_ = true*/
    {
        T tolerance_ = parameters->nonlinear_operator.newton.tolerance;
        unsigned int maximum_iterations_ = parameters->nonlinear_operator.newton.newton_max_it;
        T newton_wight_ = parameters->nonlinear_operator.newton.newton_wight;

        bool store_norms_history_ = parameters->nonlinear_operator.newton.store_norms_history;
        bool verbose_ = parameters->nonlinear_operator.newton.verbose;

        conv_newton->set_convergence_constants(tolerance_, maximum_iterations_, newton_wight_, store_norms_history_,  verbose_);
    }

    void set_newton_continuation()
/*T tolerance_, unsigned int maximum_iterations_, T newton_wight_ = T(0.8), bool store_norms_history_ = false, bool verbose_ = true*/    
    {
        T tolerance_ = parameters->deflation_continuation.newton_extended_continuation.tolerance;
        unsigned int maximum_iterations_ = parameters->deflation_continuation.newton_extended_continuation.newton_max_it;
        T newton_wight_ = parameters->deflation_continuation.newton_extended_continuation.newton_wight;

        bool store_norms_history_ = parameters->deflation_continuation.newton_extended_continuation.store_norms_history;
        bool verbose_ = parameters->deflation_continuation.newton_extended_continuation.verbose;

        T relax_tolerance_factor_ =  parameters->deflation_continuation.newton_extended_continuation.relax_tolerance_factor;

        int relax_tolerance_steps_ = parameters->deflation_continuation.newton_extended_continuation.relax_tolerance_steps;

        auto stagnation_max_l = parameters->deflation_continuation.newton_extended_continuation.stagnation_max;
        auto maximum_norm_increase_l = parameters->deflation_continuation.newton_extended_continuation.maximum_norm_increase;
        auto newton_wight_threshold_l = parameters->deflation_continuation.newton_extended_continuation.newton_wight_threshold;
        continuate->set_newton(tolerance_, maximum_iterations_, relax_tolerance_factor_, relax_tolerance_steps_, newton_wight_, store_norms_history_, verbose_, stagnation_max_l, maximum_norm_increase_l, newton_wight_threshold_l);
        detail::set_allow_knot_interpolation_failure_if_available(
            continuate,
            parameters->deflation_continuation.restart_policy.allow_knot_interpolation_failure);

    }

    void set_newton_deflation()
/*T tolerance_, unsigned int maximum_iterations_, T newton_wight_ = T(0.5), bool store_norms_history_ = false, bool verbose_ = true*/    
    {
        
        T tolerance_ = parameters->deflation_continuation.newton_extended_deflation.tolerance;
        unsigned int maximum_iterations_ = parameters->deflation_continuation.newton_extended_deflation.newton_max_it;
        T newton_wight_ = parameters->deflation_continuation.newton_extended_deflation.newton_wight;

        bool store_norms_history_ = parameters->deflation_continuation.newton_extended_deflation.store_norms_history;
        bool verbose_ = parameters->deflation_continuation.newton_extended_deflation.verbose;
        deflate->set_newton(tolerance_, maximum_iterations_, newton_wight_, store_norms_history_, verbose_);

    }

    void set_steps()
/*unsigned int max_S_, T ds_0_, unsigned int deflation_attempts_ = 5, unsigned int attempts_0_ = 4, int initial_direciton_ = -1, T step_ds_m_ = 0.2, T step_ds_p_ = 0.01*/    
    {
        
        unsigned int max_S_ = parameters->deflation_continuation.continuation_steps;
        T ds_0_ = parameters->deflation_continuation.step_size;
        T ds_max_ = parameters->deflation_continuation.max_step_size;
        int initial_direciton_ = parameters->deflation_continuation.initial_direciton;
        T step_ds_m_ = parameters->deflation_continuation.step_ds_m;
        T step_ds_p_ = parameters->deflation_continuation.step_ds_p;
        unsigned int attempts_0_ = parameters->deflation_continuation.continuation_fail_attempts;
        unsigned int deflation_attempts_ = parameters->deflation_continuation.deflation_attempts;


        deflate->set_max_retries(deflation_attempts_);
        continuate->set_steps(max_S_, ds_0_, ds_max_, initial_direciton_, step_ds_m_, step_ds_p_, attempts_0_);
        continuate_analytical->set_steps(max_S_, ds_0_, ds_max_, initial_direciton_, step_ds_m_, step_ds_p_, attempts_0_);
    }

    void set_deflation_knots()
/*std::vector<T> knots_*/    
    {
        knots->add_element(parameters->deflation_continuation.deflation_knots);
    }

    void set_branch_intersection_policy()
    {
        container::branch_intersection_policy<T> policy;
        const auto& params = parameters->deflation_continuation.branch_intersection_policy;
        policy.enabled = params.enabled;
        policy.signature_norm_index = params.signature_norm_index;
        policy.signature_tolerance = params.signature_tolerance;
        policy.state_tolerance = params.state_tolerance;
        policy.minimum_step_fraction_from_start = params.minimum_step_fraction_from_start;
        policy.verbose = params.verbose;

        if(!policy.enabled)
        {
            continuate->set_branch_intersection_checker({});
            continuate_analytical->set_branch_intersection_checker({});
            return;
        }

        auto workspace = std::make_shared<branch_distance_workspace>(vec_ops, sol_storage_def);
        auto checker =
            [this, policy, workspace](
                const T& lambda_left,
                const T_vec& x_left,
                const T& lambda_right,
                const T_vec& x_right,
                T& hit_lambda,
                T_vec& hit_x,
                std::string& reason) -> bool
            {
                container::branch_intersection_result<T> result;
                auto distance =
                    [workspace](const T_vec& a, const T_vec& b) -> T
                    {
                        return workspace->distance(a, b);
                    };
                const bool found = bif_diag->find_branch_intersection(
                    lambda_left,
                    x_left,
                    lambda_right,
                    x_right,
                    policy,
                    hit_x,
                    result,
                    distance);
                if(!found)
                {
                    if(suppress_analytical_branch_endpoint)
                    {
                        return false;
                    }
                    nonlin_op->exact_solution(lambda_right, hit_x);
                    const T exact_distance = workspace->distance(x_right, hit_x);
                    if(exact_distance <= policy.state_tolerance)
                    {
                        hit_lambda = lambda_right;
                        reason = "analytical branch endpoint";
                        if(policy.verbose)
                        {
                            log->info_f(
                                "MAIN:deflation_continuation: analytical branch endpoint detected at lambda = %le: state distance = %le, tolerance = %le.",
                                double(hit_lambda),
                                double(exact_distance),
                                double(policy.state_tolerance));
                        }
                        return true;
                    }
                    return false;
                }

                hit_lambda = result.lambda;
                std::ostringstream stream;
                stream << "known branch intersection with curve " << result.curve_number
                       << ", segment " << result.segment_id
                       << ", state distance = " << result.state_distance
                       << ", tolerance = " << result.state_tolerance;
                reason = stream.str();
                if(policy.verbose)
                {
                    log->info_f(
                        "MAIN:deflation_continuation: branch intersection detected at lambda = %le: curve = %i, segment = %llu, state distance = %le, tolerance = %le.",
                        double(result.lambda),
                        result.curve_number,
                        static_cast<unsigned long long>(result.segment_id),
                        double(result.state_distance),
                        double(result.state_tolerance));
                }
                return true;
            };

        continuate->set_branch_intersection_checker(checker);
        continuate_analytical->set_branch_intersection_checker(checker);
    }

    void use_analytical_solution(bool analytical_solution_ = false)
    {
        analytical_solution = analytical_solution_;
    }

    void add_solution_curve(const T_vec& x0_, const T& lambda0_)
    {
        bif_diag_curve_t* bdf;
        T_vec x0_stabilized;
        vec_ops->init_vector(x0_stabilized);
        vec_ops->start_use_vector(x0_stabilized);
        vec_ops->assign(x0_, x0_stabilized);
        detail::stabilize_solution_if_available(sol_storage_def, x0_stabilized);
        bif_diag->init_new_curve();
        bif_diag->get_current_ref(bdf);
        const bool old_suppress_analytical_branch_endpoint = suppress_analytical_branch_endpoint;
        suppress_analytical_branch_endpoint = true;
        try
        {
            continuate->continuate_curve(bdf, x0_stabilized, lambda0_);
        }
        catch(...)
        {
            suppress_analytical_branch_endpoint = old_suppress_analytical_branch_endpoint;
            throw;
        }
        suppress_analytical_branch_endpoint = old_suppress_analytical_branch_endpoint;
        bif_diag->close_curve();
        vec_ops->stop_use_vector(x0_stabilized);
        vec_ops->free_vector(x0_stabilized);
    }


    bool load_data(const std::string& file_name_ = {})
    {
        bool file_exists = false;
        if(!file_name_.empty())
        {
            std::ifstream load_file( (project_dir + file_name_).c_str() );
            if(load_file.good())
            {
                log->info_f("MAIN:deflation_continuation: reading data for the bifurcaiton diagram from %s ...", (project_dir + file_name_).c_str() );
                data_input ia(load_file);
                try
                {
                    ia >> (*bif_diag);
                    bif_diag->reset_curve_output_directories();
                }
                catch(const boost::archive::archive_exception& e)
                {
                    log->error_f("MAIN:deflation_continuation: reading boost serialization failed: %s", e.what() );
                }
                load_file.close();
                log->info_f("MAIN:deflation_continuation: read data for the bifurcaiton diagram from %s", (project_dir + file_name_).c_str() );
                file_exists = true;
            }
            else
            {
                log->warning_f("MAIN:deflation_continuation: failed to load saved data for the bifurcaiton diagram %s", (project_dir + file_name_).c_str() );
                file_exists = false;
            }
        }
        return file_exists;
    }

    void save_data(const std::string& file_name_ = {})
    {
        if(!file_name_.empty())
        {
            log->info_f("MAIN:deflation_continuation: saving data for the bifurcaiton diagram in %s ...", (project_dir + file_name_).c_str() );
            std::ofstream save_file( (project_dir + file_name_).c_str() );
            data_output oa(save_file);
            try
            {
                oa << (*bif_diag);
            }
            catch(const boost::archive::archive_exception& e)
            {
                log->error_f("MAIN:deflation_continuation: writing boost serialization failed: %s", e.what() );
            }
            save_file.close();
            log->info_f("MAIN:deflation_continuation: saved data for the bifurcaiton diagram in %s", (project_dir + file_name_).c_str() );
        }        
    }

    void edit()
    {
        std::string file_name = parameters->bifurcaiton_diagram_file_name;
        bool file_exists = load_data( file_name );
        if(file_exists)
        {
            std::cout << "entering interactive edit mode" << std::endl;
            std::cout << "enter 'd' to pop_back() the curve or 'q' to quit." << std::endl;
            char c = 'c';
            while(c != 'q')
            {
                std::cout << "file " << file_name << " contains:" << std::endl;
                bif_diag->print_curves_status();
                c = std::cin.get();
                if(c=='d')
                {
                    bif_diag->pop_back_curve();
                }
            }
            c = std::cin.get();
            std::cout << "save file(y/n)>>>";
            c = std::cin.get();
            if(c == 'y')
                save_data(file_name);

        }
        else
        {
            log->warning_f("MAIN:deflation_continuation: file %s doesn't exist; called edit with no file provided!", file_name.c_str());
        }

    }

    void reset_known_solutions_at_lambda(const T& lambda)
    {
        T_vec exact_solution;
        sol_storage_def->clear();
        vec_ops->init_vector(exact_solution);
        vec_ops->start_use_vector(exact_solution);
        nonlin_op->exact_solution(lambda, exact_solution);
        sol_storage_def->set_known_solution(exact_solution);
        vec_ops->stop_use_vector(exact_solution);
        vec_ops->free_vector(exact_solution);
    }

    intersection_status_t rebuild_intersections_at_lambda(const T& lambda)
    {
        reset_known_solutions_at_lambda(lambda);
        return bif_diag->find_intersection(lambda, sol_storage_def);
    }

    std::string knot_registry_file_name() const
    {
        const auto& file_name = parameters->deflation_continuation.restart_policy.knot_relocation.registry_file;
        if(file_name.empty())
        {
            return {};
        }
        if(!file_name.empty() && file_name.front() == '/')
        {
            return file_name;
        }
        return project_dir + file_name;
    }

    std::pair<T, T> relocation_bounds(const T& requested_lambda) const
    {
        std::vector<T> values;
        values.reserve(parameters->deflation_continuation.deflation_knots.size());
        for(const auto& value: parameters->deflation_continuation.deflation_knots)
        {
            values.push_back(static_cast<T>(value));
        }
        if(values.empty())
        {
            return {requested_lambda, requested_lambda};
        }
        std::sort(values.begin(), values.end());
        values.erase(std::unique(values.begin(), values.end()), values.end());
        if(values.size() == 1)
        {
            return {values.front(), values.front()};
        }

        auto same_lambda = [](const T& a, const T& b)
        {
            const auto scalar_abs = [](const T& value)
            {
                return value < T(0) ? -value : value;
            };
            const T scale = std::max<T>(T(1), std::max<T>(scalar_abs(a), scalar_abs(b)));
            return scalar_abs(a - b) <= T(64)*std::numeric_limits<T>::epsilon()*scale;
        };

        for(std::size_t i = 0; i < values.size(); ++i)
        {
            if(same_lambda(values[i], requested_lambda))
            {
                const T lower = (i == 0) ? values[i] : values[i - 1];
                const T upper = (i + 1 >= values.size()) ? values[i] : values[i + 1];
                return {lower, upper};
            }
        }

        auto upper_it = std::upper_bound(values.begin(), values.end(), requested_lambda);
        if(upper_it == values.begin())
        {
            return {values.front(), values.front()};
        }
        if(upper_it == values.end())
        {
            return {values.back(), values.back()};
        }
        return {*(upper_it - 1), *upper_it};
    }

    bool candidate_inside_bounds(const T& candidate, const std::pair<T, T>& bounds) const
    {
        return candidate > bounds.first && candidate < bounds.second;
    }

    T relocation_candidate(const T& requested_lambda, const unsigned int candidate_index) const
    {
        const auto& settings = parameters->deflation_continuation.restart_policy.knot_relocation;
        const unsigned int radius_count = std::max(1u, (settings.candidate_count + 1u)/2u);
        const unsigned int radius_index = candidate_index/2u;
        const T alpha = radius_count == 1u
            ? T(0)
            : static_cast<T>(radius_index)/static_cast<T>(radius_count - 1u);
        const T radius = settings.min_shift_abs + alpha*(settings.max_shift_abs - settings.min_shift_abs);
        const bool positive_slot = (candidate_index%2u) == 0u;
        const T sign = (positive_slot == settings.prefer_positive_shift) ? T(1) : T(-1);
        return requested_lambda + sign*radius;
    }

    bool acceptable_relocated_intersection(
        const intersection_status_t& status,
        const unsigned int required_intersections) const
    {
        if(!status.ok())
        {
            return false;
        }
        if(parameters->deflation_continuation.restart_policy.knot_relocation.require_all_intersections &&
           status.added < required_intersections)
        {
            return false;
        }
        return true;
    }

    bool relocate_knot(
        const T& requested_lambda,
        const intersection_status_t& failed_status,
        knot_registry_t& registry,
        T& effective_lambda,
        intersection_status_t& effective_status)
    {
        const auto& settings = parameters->deflation_continuation.restart_policy.knot_relocation;
        if(!settings.enabled || settings.candidate_count == 0)
        {
            return false;
        }

        const unsigned int required_intersections =
            failed_status.added + failed_status.failed + failed_status.missing_data;
        const auto bounds = relocation_bounds(requested_lambda);
        for(unsigned int candidate_index = 0; candidate_index < settings.candidate_count; ++candidate_index)
        {
            const T candidate = relocation_candidate(requested_lambda, candidate_index);
            if(!candidate_inside_bounds(candidate, bounds))
            {
                continue;
            }
            const auto status = rebuild_intersections_at_lambda(candidate);
            if(acceptable_relocated_intersection(status, required_intersections))
            {
                effective_lambda = candidate;
                effective_status = status;
                registry.set(
                    requested_lambda,
                    effective_lambda,
                    "intersection_newton_failed_at_requested_knot",
                    effective_status);
                if(settings.save_registry)
                {
                    registry.save();
                }
                log->warning_f(
                    "MAIN:deflation_continuation: relocated requested knot %le to non-singular knot %le after restart intersection failure; added = %u, skipped_discontinuous = %u.",
                    double(requested_lambda),
                    double(effective_lambda),
                    effective_status.added,
                    effective_status.skipped_discontinuous);
                return true;
            }
        }
        return false;
    }

    bool candidate_inside_active_interval(
        const T& candidate,
        const T& lambda_left,
        const T& lambda_right) const
    {
        const T lower = std::min(lambda_left, lambda_right);
        const T upper = std::max(lambda_left, lambda_right);
        return candidate > lower && candidate < upper;
    }

    bool relocate_active_continuation_knot(
        const T& requested_lambda,
        const T& lambda_left,
        const T_vec& x_left,
        const T& lambda_right,
        const T_vec& x_right,
        knot_registry_t& registry,
        T& effective_lambda,
        T_vec& effective_x)
    {
        const auto& settings = parameters->deflation_continuation.restart_policy.knot_relocation;
        if(!settings.enabled || settings.candidate_count == 0)
        {
            return false;
        }
        if(lambda_right == lambda_left)
        {
            return false;
        }

        const auto bounds = relocation_bounds(requested_lambda);
        for(unsigned int candidate_index = 0; candidate_index < settings.candidate_count; ++candidate_index)
        {
            const T candidate = relocation_candidate(requested_lambda, candidate_index);
            if(!candidate_inside_bounds(candidate, bounds) ||
               !candidate_inside_active_interval(candidate, lambda_left, lambda_right))
            {
                continue;
            }

            const T w = (candidate - lambda_left)/(lambda_right - lambda_left);
            vec_ops->assign_mul(T(1) - w, x_left, w, x_right, effective_x);
            detail::stabilize_solution_if_available(sol_storage_def, effective_x);

            const bool converged = newton->solve(nonlin_op, effective_x, candidate);
            if(!converged)
            {
                log->info_f(
                    "MAIN:deflation_continuation: active knot relocation candidate %le for requested knot %le failed Newton interpolation.",
                    double(candidate),
                    double(requested_lambda));
                continue;
            }

            detail::stabilize_solution_if_available(sol_storage_def, effective_x);
            effective_lambda = candidate;

            intersection_status_t status;
            status.added = 1;
            registry.set(
                requested_lambda,
                effective_lambda,
                "active_continuation_interpolation_failed_at_requested_knot",
                status);
            if(settings.save_registry)
            {
                registry.save();
            }

            log->warning_f(
                "MAIN:deflation_continuation: shifted active continuation knot %le to validated non-singular knot %le after interpolation Newton failure.",
                double(requested_lambda),
                double(effective_lambda));
            return true;
        }

        log->warning_f(
            "MAIN:deflation_continuation: active continuation knot relocation failed for requested knot %le inside step [%le, %le].",
            double(requested_lambda),
            double(lambda_left),
            double(lambda_right));
        return false;
    }


    void execute()
    {
        std::string file_name = parameters->bifurcaiton_diagram_file_name;
        //Algorithm pseudocode:
        //
        //Set second knot value: knots.next()
        //while(true)
        //{
        //  knot_value = knot.get_value()
        //  perform deflation until a new solutoin is found
        //  if the solution is found:
        //    create a new curve and continuate it until it is finished
        //    add interseciton of the curve with the current knot value to the deflator: deflation_container.push_back()
        //  else 
        //    if(!knot.next())
        //      break
        //    else
        //      deflation_container.clear()
        //}
        //

        bool file_exists = load_data(file_name);
        knot_registry_t knot_registry(knot_registry_file_name());
        auto knot_resolver = [&knot_registry](const T& requested_lambda, T& effective_lambda) -> bool
        {
            typename knot_registry_t::entry registry_entry;
            if(knot_registry.find(requested_lambda, registry_entry))
            {
                effective_lambda = registry_entry.effective;
                return true;
            }
            effective_lambda = requested_lambda;
            return false;
        };
        auto active_knot_relocator =
            [this, &knot_registry](
                const T& requested_lambda,
                const T& lambda_left,
                const T_vec& x_left,
                const T& lambda_right,
                const T_vec& x_right,
                T& effective_lambda,
                T_vec& effective_x) -> bool
            {
                return relocate_active_continuation_knot(
                    requested_lambda,
                    lambda_left,
                    x_left,
                    lambda_right,
                    x_right,
                    knot_registry,
                    effective_lambda,
                    effective_x);
            };
        detail::set_knot_resolver_if_available(continuate, knot_resolver);
        detail::set_knot_relocator_if_available(continuate, active_knot_relocator);
        detail::set_knot_resolver_if_available(continuate_analytical, knot_resolver);
        detail::set_knot_relocator_if_available(continuate_analytical, active_knot_relocator);

        // force file printing skip after serialization
        // unless it is done, the skip data is taken from the serialization class from file!
        bif_diag->set_skip_output(skip_files);


        T_vec x_deflation; //pointer to the found deflated solution
        bool is_there_a_next_knot = knots->next();

        //perform analytical solution continuation if desired and if it is the first run
        if( (analytical_solution)&&(!file_exists))
        {
            log->info("MAIN:deflation_continuation: using the analytical solution to form a curve...");
            bif_diag_curve_t* bdf;
            T lambda = knots->get_value();
            vec_ops->init_vector(x_deflation); vec_ops->start_use_vector(x_deflation);
            nonlin_op->exact_solution(lambda, x_deflation);
            detail::stabilize_solution_if_available(sol_storage_def, x_deflation);
            bif_diag->init_new_curve();
            bif_diag->get_current_ref(bdf);
            const bool analytical_success = continuate_analytical->continuate_curve(bdf, x_deflation, lambda);
            bif_diag->close_curve();
            if(analytical_success || parameters->deflation_continuation.restart_policy.allow_failed_continuation_curve_save)
            {
                save_data(file_name);
            }
            else
            {
                log->warning("MAIN:deflation_continuation: analytical curve continuation failed; discarding curve according to restart policy.");
                bif_diag->discard_current_curve();
            }
            vec_ops->stop_use_vector(x_deflation); vec_ops->free_vector(x_deflation);
            log->info("MAIN:deflation_continuation: analytical solution formed.");
        }
        //

        int number_of_solutions = 0;
        rejected_candidate_cache rejected_candidates(vec_ops);
        T rejected_cache_lambda = T(0);
        bool rejected_cache_active = false;
        unsigned int failed_continuations_at_knot = 0;
        while(is_there_a_next_knot)
        {
            
            const T requested_lambda = knots->get_value();
            T lambda = requested_lambda;
            if(parameters->deflation_continuation.restart_policy.knot_relocation.enabled)
            {
                typename knot_registry_t::entry registry_entry;
                if(knot_registry.find(requested_lambda, registry_entry))
                {
                    lambda = registry_entry.effective;
                    log->warning_f(
                        "MAIN:deflation_continuation: requested knot %le is mapped to validated non-singular knot %le from %s.",
                        double(requested_lambda),
                        double(lambda),
                        knot_registry_file_name().c_str());
                }
            }
            log->info_f("MAIN:deflation_continuation: currently having %i curves.", bif_diag->current_curve() );
            
            auto intersection_status = rebuild_intersections_at_lambda(lambda);
            const bool relocation_needed =
                !intersection_status.ok() &&
                parameters->deflation_continuation.restart_policy.knot_relocation.enabled;
            if(relocation_needed)
            {
                T relocated_lambda = lambda;
                intersection_status_t relocated_status;
                if(relocate_knot(requested_lambda, intersection_status, knot_registry, relocated_lambda, relocated_status))
                {
                    lambda = relocated_lambda;
                    intersection_status = relocated_status;
                }
                else
                {
                    intersection_status = rebuild_intersections_at_lambda(lambda);
                }
            }
            const bool intersections_incomplete = !intersection_status.ok();
            if(intersections_incomplete && !parameters->deflation_continuation.restart_policy.allow_incomplete_restart_intersections)
            {
                log->warning_f(
                    "MAIN:deflation_continuation: skipping deflation at requested lambda = %lf, effective lambda = %lf because restart intersections are incomplete: added = %u, failed = %u, missing_data = %u, skipped_discontinuous = %u.",
                    double(requested_lambda),
                    double(lambda),
                    intersection_status.added,
                    intersection_status.failed,
                    intersection_status.missing_data,
                    intersection_status.skipped_discontinuous);
                is_there_a_next_knot = knots->next();
                continue;
            }
            if(intersections_incomplete)
            {
                log->warning_f(
                    "MAIN:deflation_continuation: continuing with incomplete restart intersections at requested lambda = %lf, effective lambda = %lf because policy allows it: added = %u, failed = %u, missing_data = %u, skipped_discontinuous = %u.",
                    double(requested_lambda),
                    double(lambda),
                    intersection_status.added,
                    intersection_status.failed,
                    intersection_status.missing_data,
                    intersection_status.skipped_discontinuous);
            }

            if(!rejected_cache_active || !same_parameter_value(lambda, rejected_cache_lambda))
            {
                rejected_candidates.clear();
                rejected_cache_lambda = lambda;
                rejected_cache_active = true;
                failed_continuations_at_knot = 0;
            }

            const unsigned int max_failed_continuations =
                parameters->deflation_continuation.restart_policy.max_failed_continuations_per_knot;
            if(max_failed_continuations > 0 &&
               failed_continuations_at_knot >= max_failed_continuations)
            {
                log->warning_f(
                    "MAIN:deflation_continuation: skipping requested lambda = %lf, effective lambda = %lf after %u failed continuation attempts at this knot.",
                    double(requested_lambda),
                    double(lambda),
                    failed_continuations_at_knot);
                is_there_a_next_knot = knots->next();
                continue;
            }

            bool is_new_solution = false;
            bool candidate_duplicate = false;
            unsigned int duplicate_retry = 0;
            const unsigned int duplicate_retry_max =
                parameters->deflation_continuation.restart_policy.duplicate_after_deflation_retries;
            do
            {
                candidate_duplicate = false;
                is_new_solution = deflate->find_solution(lambda);
                if(is_new_solution)
                {
                    deflate->get_solution_ref(x_deflation);
                    detail::stabilize_solution_if_available(sol_storage_def, x_deflation);
                    T nearest_rejected_distance = std::numeric_limits<T>::infinity();
                    const bool rejected_distance_available =
                        rejected_candidates.nearest_distance(lambda, x_deflation, nearest_rejected_distance);
                    if(rejected_distance_available &&
                       nearest_rejected_distance <= parameters->deflation_continuation.restart_policy.failed_continuation_rejection_tolerance)
                    {
                        candidate_duplicate = true;
                        is_new_solution = false;
                        log->warning_f(
                            "MAIN:deflation_continuation: deflated Newton returned a candidate rejected after failed continuation at lambda = %lf with stabilized distance = %le and tolerance = %le.",
                            double(lambda),
                            double(nearest_rejected_distance),
                            double(parameters->deflation_continuation.restart_policy.failed_continuation_rejection_tolerance));
                    }
                    if(!candidate_duplicate && parameters->deflation_continuation.restart_policy.check_duplicate_after_deflation)
                    {
                        T nearest_distance = std::numeric_limits<T>::infinity();
                        const bool distance_available =
                            detail::nearest_stabilized_distance_if_available(sol_storage_def, x_deflation, nearest_distance);
                        if(distance_available &&
                           nearest_distance <= parameters->deflation_continuation.restart_policy.duplicate_after_deflation_tolerance)
                        {
                            candidate_duplicate = true;
                            is_new_solution = false;
                            log->warning_f(
                                "MAIN:deflation_continuation: deflated Newton returned a duplicate solution at lambda = %lf with stabilized distance = %le and tolerance = %le.",
                                double(lambda),
                                double(nearest_distance),
                                double(parameters->deflation_continuation.restart_policy.duplicate_after_deflation_tolerance));
                        }
                    }
                }
                duplicate_retry++;
            }
            while(candidate_duplicate && duplicate_retry <= duplicate_retry_max);

            if(is_new_solution)
            {
                number_of_solutions++;
                log->info_f("MAIN:deflation_continuation: found %i solutions for lambda = %lf.", number_of_solutions, double(lambda) );
                
                bif_diag_curve_t* bdf;
                
                bif_diag->init_new_curve();
                bif_diag->get_current_ref(bdf);
                //std::cin.get(); 
                const bool continuation_success = continuate->continuate_curve(bdf, x_deflation, lambda);
                bif_diag->close_curve();
                //std::cin.get(); 
                if(continuation_success || parameters->deflation_continuation.restart_policy.allow_failed_continuation_curve_save)
                {
                    save_data(file_name);
                    bdf->find_intersection(lambda, sol_storage_def);
                }
                else
                {
                    log->warning_f(
                        "MAIN:deflation_continuation: continuation of a new curve at lambda = %lf failed; discarding curve according to restart policy.",
                        double(lambda));
                    bif_diag->discard_current_curve();
                    rejected_candidates.add(lambda, x_deflation);
                    failed_continuations_at_knot++;
                    log->warning_f(
                        "MAIN:deflation_continuation: stored failed-continuation candidate rejection %u/%u at lambda = %lf.",
                        failed_continuations_at_knot,
                        max_failed_continuations,
                        double(lambda));
                    save_data(file_name);
                    if(max_failed_continuations > 0 &&
                       failed_continuations_at_knot >= max_failed_continuations)
                    {
                        log->warning_f(
                            "MAIN:deflation_continuation: reached max_failed_continuations_per_knot = %u at requested lambda = %lf, effective lambda = %lf; advancing to the next knot.",
                            max_failed_continuations,
                            double(requested_lambda),
                            double(lambda));
                        is_there_a_next_knot = knots->next();
                    }
                }
                
            }
            else
            {
                is_there_a_next_knot = knots->next();
            }
        }
    }



private:
//  references to the external classes:
    VectorOperations* vec_ops;
    VectorFileOperations* file_ops;
    Log* log;
    Log* log_linsolver;
    NonlinearOperations* nonlin_op;
    Parameters* parameters;

//created locally:
    LinearOperator* lin_op;
    Preconditioner* precond;
    monitor_t* mon;
    monitor_t* mon_orig; 
    sherman_morrison_linear_system_solve_t* SM;
    convergence_newton_t* conv_newton;
    system_operator_t* system_operator;
    newton_t* newton;
    knots_t* knots;
    bif_diag_t* bif_diag = nullptr;
    continuate_t* continuate;
    continuate_analytical_t* continuate_analytical;
    deflate_t* deflate;
    sol_storage_def_t* sol_storage_def = nullptr;
    bool owns_solution_storage = true;
    bool suppress_analytical_branch_endpoint = false;
    std::string project_dir;
    bool analytical_solution = false;
    unsigned int skip_files;
};


}

#endif // __DEFLATION_CONTINUATION_HPP__
