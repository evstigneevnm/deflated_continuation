#ifndef __CONTINUATION_HPP__
#define __CONTINUATION_HPP__

/**
*    The main part of the deflation-continuation process.
*
*    Continuation class that utilizes single step advance to continue the solution 
*    until it returns back or reaches two boundaries of min and max knots values.
*
*/

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include <numerical_algos/newton_solvers/newton_solver_extended.h>

#include <continuation/predictor_adaptive.h>
#include <continuation/system_operator_continuation.h>
#include <continuation/advance_solution.h>
#include <continuation/chart_helpers.h>
#include <continuation/initial_tangent.h>
#include <continuation/convergence_strategy.h>
#include <continuation/continuation_endpoint_state.h>
#include <continuation/continuation_result.h>
#include <continuation/observational_knot_sample.h>
#include <continuation/pending_branch_event.h>
#include <continuation/progress_monitor.h>
#include <continuation/semicurve_tangent_cache.h>
#include <containers/branch_intersection.h>
#include <containers/curve_endpoint_reason.h>



namespace continuation
{

namespace detail
{

template<class Curve>
auto start_new_curve_segment_if_available(Curve* curve) -> decltype(curve->start_new_segment(), void())
{
    curve->start_new_segment();
}

template<class Curve>
auto current_curve_segment_id_if_available(Curve* curve, int)
    -> decltype(curve->get_current_segment_id(), uint64_t())
{
    return curve->get_current_segment_id();
}

inline uint64_t current_curve_segment_id_if_available(...)
{
    return 0;
}

inline void start_new_curve_segment_if_available(...)
{
}

template<class Object>
auto set_verbose_if_available(Object* object, const bool value, int)
    -> decltype(object->set_verbose(value), void())
{
    object->set_verbose(value);
}

inline void set_verbose_if_available(...)
{
}

template<class Curve, class Event>
auto record_symmetry_intersection_if_available(Curve* curve, const Event& event, int)
    -> decltype(curve->record_symmetry_intersection(event), void())
{
    curve->record_symmetry_intersection(event);
}

template<class Curve, class Event>
auto record_symmetry_intersection_if_available(Curve* curve, const Event& event, long)
    -> decltype(
        curve->record_symmetry_intersection(
            event.previous_order,
            event.candidate_order,
            event.previous_transverse_ratio,
            event.candidate_transverse_ratio,
            event.refinements),
        void())
{
    curve->record_symmetry_intersection(
        event.previous_order,
        event.candidate_order,
        event.previous_transverse_ratio,
        event.candidate_transverse_ratio,
        event.refinements);
}

inline void record_symmetry_intersection_if_available(...)
{
}

}

template<class VectorOperations, class VectorFileOperations, class Log, class NonlinearOperator, class LinearOperator,  class Knots, class LinearSolver, class Newton, class Curve, template<class, class, class, class, class> class SystemOperatorContinuation = system_operator_continuation>
class continuation
{
protected:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;
    typedef container::curve_endpoint_reason endpoint_reason_t;

private:
    typedef std::pair<bool, bool> bools2;
    typedef std::function<bool(const T& requested_lambda, T& effective_lambda)> knot_resolver_t;
    typedef std::function<bool(
        const T& requested_lambda,
        const T& lambda_left,
        const T_vec& x_left,
        const T& lambda_right,
        const T_vec& x_right,
        T& effective_lambda,
        T_vec& effective_x)> knot_relocator_t;
    typedef std::function<container::branch_intersection_detection(
        const T& lambda_left,
        const T_vec& x_left,
        const T& lambda_right,
        const T_vec& x_right,
        T& hit_lambda,
        T_vec& hit_x,
        int& hit_curve_number,
        uint64_t& hit_segment_id,
        T& hit_forward_steps_ahead,
        T& hit_endpoint_distance_step_ratio,
        container::curve_provenance& hit_target_provenance,
        std::string& reason)> branch_intersection_checker_t;
    typedef std::function<bool(
        Curve* curve,
        const T& lambda_left,
        const T_vec& x_left,
        const T& lambda_right,
        const T_vec& x_right,
        T& hit_lambda,
        T_vec& hit_x,
        std::string& reason)> self_intersection_checker_t;


    typedef SystemOperatorContinuation<
        VectorOperations, 
        NonlinearOperator,
        LinearOperator,
        LinearSolver,
        Log
        > system_operator_cont_t;

    typedef newton_method_extended::convergence_strategy<
        VectorOperations, 
        NonlinearOperator, 
        Log> convergence_newton_cont_t;

    typedef numerical_algos::newton_method_extended::newton_solver_extended<
        VectorOperations, 
        NonlinearOperator,
        system_operator_cont_t, 
        convergence_newton_cont_t, 
        T /* point solution class here instead of real!*/ 
        > newton_cont_t;

    typedef predictor_adaptive<
        VectorOperations,
        Log
        > predictor_cont_t;

    typedef advance_solution<
        VectorOperations,
        Log,
        newton_cont_t,
        Newton,
        NonlinearOperator,
        system_operator_cont_t,
        predictor_cont_t,
        convergence_newton_cont_t
        >advance_step_cont_t;

    typedef initial_tangent<
        VectorOperations,
        Log,
        Newton,
        NonlinearOperator, 
        LinearOperator,
        LinearSolver
        > tangent_0_cont_t;

    using seed_tangent_cache_t = semicurve_tangent_cache<VectorOperations>;




public:
    continuation(VectorOperations* vec_ops_, VectorFileOperations* file_ops_, Log* log_, NonlinearOperator* nonlin_op_, LinearOperator* lin_op_, Knots* knots_, LinearSolver* SM_, Newton* newton_):
    vec_ops(vec_ops_),
    file_ops(file_ops_),
    log(log_),
    nonlin_op(nonlin_op_),
    knots(knots_),
    SM(SM_),
    newton(newton_),
    lin_op(lin_op_),
    seed_tangent_cache(vec_ops_)
    {
        predict = new predictor_cont_t(vec_ops, log);
        system_operator_cont = new system_operator_cont_t(vec_ops, log, lin_op, SM);
        conv_newton_cont = new convergence_newton_cont_t(vec_ops, log);
        newton_cont = new newton_cont_t(vec_ops, system_operator_cont, conv_newton_cont);
        continuation_step = new advance_step_cont_t(vec_ops, log, system_operator_cont, newton_cont, newton, predict, conv_newton_cont);
        init_tangent = new tangent_0_cont_t(vec_ops, log, newton, lin_op, SM);


        max_S = 100;
        set_all_vectors();
    }
    ~continuation()
    {
        clear_recovery_checkpoints();
        unset_all_vectors();
        delete init_tangent;
        delete continuation_step;
        delete newton_cont;
        delete conv_newton_cont;
        delete system_operator_cont;
        delete predict;

    }

    void set_steps(unsigned int max_S_, T ds_0_, T ds_max_, int initial_direciton_ = -1, T step_ds_m_ = 0.01, T step_ds_p_ = 0.01, unsigned int attempts_0_ = 4)
    {
        max_S = max_S_;
        initial_direciton = initial_direciton_;
        predict->set_steps(ds_0_, ds_max_, step_ds_m_, step_ds_p_, attempts_0_);
        accepted_progress_monitor.configure(progress_policy, ds_0_);
        
    }

    void set_steps(
        const unsigned int max_S_,
        const T ds_0_,
        const T ds_max_,
        const int initial_direciton_,
        const corrector_retry_policy<T>& retry_policy)
    {
        max_S = max_S_;
        initial_direciton = initial_direciton_;
        predict->set_steps(ds_0_, ds_max_, retry_policy);
        accepted_progress_monitor.configure(progress_policy, ds_0_);
    }

    void set_progress_monitor_policy(const progress_monitor_policy<T>& policy)
    {
        progress_policy = policy;
        accepted_progress_monitor.configure(progress_policy, predict->get_initial_ds());
    }

    void set_predictor_chart_policy(const predictor_chart_policy<T>& policy)
    {
        init_tangent->set_predictor_chart_policy(policy);
        continuation_step->set_predictor_chart_policy(policy);
    }

    void set_isotropy_transition_policy(
        const symmetry::continuation::isotropy_transition_policy<T>& policy)
    {
        continuation_step->set_isotropy_transition_policy(policy);
    }

    void set_newton(T tolerance_, unsigned int maximum_iterations_, T relax_tolerance_factor_, int relax_tolerance_steps_, T newton_wight_ = T(1), bool store_norms_history_ = false, bool verbose_ = true, unsigned int stagnation_max_p = 10, T maximum_norm_increase_p = 0.1, T newton_wight_threshold_p = 1.0e-12)
    {
        conv_newton_cont->set_convergence_constants(tolerance_, maximum_iterations_, relax_tolerance_factor_, relax_tolerance_steps_, newton_wight_, store_norms_history_,  verbose_, stagnation_max_p, maximum_norm_increase_p, newton_wight_threshold_p);
        continuation_step->set_verbose(verbose_);
        detail::set_verbose_if_available(system_operator_cont, verbose_, 0);
        epsilon = T(100.0)*tolerance_; //tolerance to check distance between vectors in curves.
    }

    void set_solution_postprocessor(std::function<void(T_vec&)> solution_postprocessor_)
    {
        solution_postprocessor = std::move(solution_postprocessor_);
    }

    void set_knot_resolver(knot_resolver_t knot_resolver_)
    {
        knot_resolver = std::move(knot_resolver_);
    }

    void set_knot_relocator(knot_relocator_t knot_relocator_)
    {
        knot_relocator = std::move(knot_relocator_);
    }

    void set_allow_knot_interpolation_failure(const bool allow_)
    {
        allow_knot_interpolation_failure = allow_;
    }

    void set_parameter_bounds(
        const T minimum,
        const T maximum)
    {
        if(!(minimum < maximum))
        {
            throw std::invalid_argument(
                "continuation::set_parameter_bounds requires minimum < maximum");
        }
        configured_lambda_min = minimum;
        configured_lambda_max = maximum;
        parameter_bounds_set = true;
    }

    void clear_parameter_bounds()
    {
        parameter_bounds_set = false;
    }

    void set_preserve_last_converged_boundary_point(
        const bool preserve)
    {
        preserve_last_converged_boundary_point = preserve;
    }

    void set_branch_intersection_checker(branch_intersection_checker_t checker_)
    {
        branch_intersection_checker = std::move(checker_);
    }

    void set_branch_intersection_refinement(
        const T step_factor,
        const unsigned int maximum_refinements,
        const unsigned int minimum_refinements_for_verification,
        const T maximum_verified_steps_ahead,
        const T maximum_verified_distance_step_ratio,
        const bool localize_analytical_targets,
        const T analytical_target_parameter_tolerance)
    {
        if(step_factor <= T(0) || step_factor >= T(1))
        {
            throw std::invalid_argument(
                "branch intersection refinement step factor must be in (0,1)");
        }
        if(maximum_refinements == 0)
        {
            throw std::invalid_argument(
                "branch intersection maximum refinements must be positive");
        }
        if(minimum_refinements_for_verification == 0 ||
           minimum_refinements_for_verification > maximum_refinements)
        {
            throw std::invalid_argument(
                "branch intersection minimum verification refinements must be positive and not exceed the refinement limit");
        }
        if(maximum_verified_steps_ahead <= T(0) ||
           maximum_verified_distance_step_ratio <= T(0) ||
           analytical_target_parameter_tolerance <= T(0))
        {
            throw std::invalid_argument(
                "branch intersection verification thresholds must be positive");
        }
        branch_refinement_step_factor = step_factor;
        maximum_branch_refinements = maximum_refinements;
        minimum_branch_refinements_for_verification =
            minimum_refinements_for_verification;
        maximum_verified_branch_steps_ahead = maximum_verified_steps_ahead;
        maximum_verified_branch_distance_step_ratio =
            maximum_verified_distance_step_ratio;
        localize_analytical_branch_targets = localize_analytical_targets;
        analytical_branch_parameter_tolerance =
            analytical_target_parameter_tolerance;
    }

    void set_self_intersection_checker(self_intersection_checker_t checker_)
    {
        self_intersection_checker = std::move(checker_);
    }


    void update_knots()
    {
        if(parameter_bounds_set)
        {
            lambda_min = configured_lambda_min;
            lambda_max = configured_lambda_max;
            return;
        }
        lambda_min = knots->get_min_value();
        lambda_max = knots->get_max_value();
        
    }

    continuation_curve_result<T> continuate_curve_result(
        Curve*& curve_,
        const T_vec& x0_,
        const T& lambda0_)
    {
        update_knots();
        bif_diag = curve_;
        direction = initial_direciton;
        fail_flag = false;
        hard_failure = false;
        endpoint_state.reset_curve();
        last_curve_result = {};
        recovery_checkpoint_valid.fill(false);
        last_accepted_checkpoint_valid = false;
        current_curve_point_offset = 0;
        branch_event.clear();
        just_interpolated = false;
        continue_next_step = true;
        
        //make a copy here? or just use the provided reference
        //x0 = x0_, lambda0 = lambda0_;
        chart::prepare_continuation_seed(
            vec_ops,
            log,
            nonlin_op,
            x0_,
            x_start);
        vec_ops->assign(x_start, x0);
        lambda0 = lambda0_;
        lambda_start = lambda0_;
        seed_tangent_cache.clear();
        
        break_semicurve = 0;

        while (break_semicurve < 2)
        {
            const unsigned int semicurve_index =
                last_curve_result.semicurves_started;
            if(semicurve_index >= last_curve_result.semicurves.size())
            {
                break;
            }
            current_semicurve_points = 0;
            current_semicurve_endpoint_reason = endpoint_reason_t::none;
            current_semicurve_failure = continuation_failure_kind::none;
            current_semicurve_failure_message.clear();
            current_semicurve_attempted_step = T(0);
            current_semicurve_retry_count = 0;
            current_semicurve_last_parameter = lambda_start;
            last_accepted_checkpoint_valid = false;
            continue_next_step = true;
            chart::prepare_continuation_seed(
                vec_ops,
                log,
                nonlin_op,
                x_start,
                x0);
            detail::start_new_curve_segment_if_available(bif_diag);
            const std::uint64_t segment_id =
                detail::current_curve_segment_id_if_available(
                    bif_diag,
                    0);
            const std::uint64_t first_point_index =
                current_curve_point_offset;
            const int semicurve_direction = direction;
            start_semicurve();
            auto& semicurve =
                last_curve_result.semicurves[semicurve_index];
            semicurve.direction = semicurve_direction;
            semicurve.accepted_points = current_semicurve_points;
            semicurve.segment_id = segment_id;
            semicurve.first_point_index = first_point_index;
            semicurve.last_point_index = current_semicurve_points == 0
                ? first_point_index
                : first_point_index + current_semicurve_points - 1;
            semicurve.start_parameter = lambda_start;
            semicurve.last_parameter = current_semicurve_last_parameter;
            semicurve.endpoint_reason = current_semicurve_endpoint_reason;
            semicurve.failure = current_semicurve_failure;
            semicurve.attempted_step = current_semicurve_attempted_step;
            semicurve.retry_count = current_semicurve_retry_count;
            semicurve.message = current_semicurve_failure_message;
            semicurve.status =
                current_semicurve_failure != continuation_failure_kind::none ||
                container::is_incomplete_endpoint(
                    current_semicurve_endpoint_reason)
                    ? semicurve_status::open_recoverable
                    : semicurve_status::complete;
            current_curve_point_offset += current_semicurve_points;
            if(semicurve.has_progress())
            {
                capture_recovery_checkpoint(semicurve_index);
            }
            ++last_curve_result.semicurves_started;
            if(current_semicurve_endpoint_reason ==
               endpoint_reason_t::closed_return)
            {
                last_curve_result.branch_closed = true;
            }
            change_direction(); //if we reached the origin, then this is irrelevant. Else, change direction and do it again
            vec_ops->assign(x_start, x0);
            lambda0 = lambda_start;
            if(fail_flag)
            {
                log->info_f("continuation::continuate_curve: previous semicurve returned with fail flag.");
                fail_flag = false;
            }
        }
        bif_diag->print_curve();
        return last_curve_result;
    }

    bool continuate_curve(Curve*& curve_, const T_vec& x0_, const T& lambda0_)
    {
        return continuate_curve_result(curve_, x0_, lambda0_).complete();
    }

    const continuation_curve_result<T>& last_continuation_result() const
    {
        return last_curve_result;
    }

    bool copy_recovery_checkpoint(
        const unsigned int semicurve_index,
        T_vec& state,
        T_vec& tangent,
        T& parameter,
        T& parameter_tangent) const
    {
        if(semicurve_index >= recovery_checkpoint_valid.size() ||
           !recovery_checkpoint_valid[semicurve_index])
        {
            return false;
        }
        vec_ops->assign(recovery_checkpoint_state[semicurve_index], state);
        vec_ops->assign(recovery_checkpoint_tangent[semicurve_index], tangent);
        parameter = recovery_checkpoint_parameter[semicurve_index];
        parameter_tangent =
            recovery_checkpoint_parameter_tangent[semicurve_index];
        return true;
    }


protected: //changed to protected for inheritance
    //passed:
    VectorOperations* vec_ops;
    VectorFileOperations* file_ops;
    Log* log;
    NonlinearOperator* nonlin_op;
    Knots* knots;
    LinearSolver* SM;
    Newton* newton;
    LinearOperator* lin_op;
    
    //created localy:
    predictor_cont_t* predict;
    system_operator_cont_t* system_operator_cont;
    convergence_newton_cont_t *conv_newton_cont;
    newton_cont_t* newton_cont;
    advance_step_cont_t* continuation_step;
    tangent_0_cont_t* init_tangent;
    Curve* bif_diag;
    seed_tangent_cache_t seed_tangent_cache;

    int direction = -1;
    int initial_direciton = 1;
    unsigned int max_S;
    T epsilon = T(1.0e-5);


    void change_direction()
    {
        direction *= -1;
    }

    //vectors and points for continuation

    T lambda_start; T_vec x_start;
    T lambda0, lambda0_s, lambda1, lambda1_s;
    T lambda_min, lambda_max;
    T configured_lambda_min = T(0);
    T configured_lambda_max = T(0);
    T_vec x0, x0_s, x1, x1_back, x1_s, x_check, x_output, x_knot_sample, x_branch_intersection, x_pending_branch_event;
    T_vec x_last_accepted, x_last_accepted_s;
    char break_semicurve = 0;
    bool fail_flag = false;
    bool hard_failure = false;
    bool continue_next_step = true;
    bool just_interpolated = false;
    continuation_endpoint_state endpoint_state;
    bool allow_knot_interpolation_failure = false;
    bool parameter_bounds_set = false;
    bool preserve_last_converged_boundary_point = true;
    std::function<void(T_vec&)> solution_postprocessor;
    knot_resolver_t knot_resolver;
    knot_relocator_t knot_relocator;
    branch_intersection_checker_t branch_intersection_checker;
    T branch_refinement_step_factor = T(0.25);
    unsigned int maximum_branch_refinements = 8;
    unsigned int minimum_branch_refinements_for_verification = 3;
    T maximum_verified_branch_steps_ahead = T(0.1);
    T maximum_verified_branch_distance_step_ratio = T(0.3);
    bool localize_analytical_branch_targets = true;
    T analytical_branch_parameter_tolerance = T(1.0e-7);
    pending_branch_event<T> branch_event;
    self_intersection_checker_t self_intersection_checker;
    progress_monitor_policy<T> progress_policy;
    progress_monitor<T> accepted_progress_monitor;
    continuation_curve_result<T> last_curve_result;
    unsigned int current_semicurve_points = 0;
    endpoint_reason_t current_semicurve_endpoint_reason =
        endpoint_reason_t::none;
    continuation_failure_kind current_semicurve_failure =
        continuation_failure_kind::none;
    std::string current_semicurve_failure_message;
    T current_semicurve_attempted_step = T(0);
    unsigned int current_semicurve_retry_count = 0;
    T current_semicurve_last_parameter = T(0);
    std::uint64_t current_curve_point_offset = 0;
    std::array<T_vec, 2> recovery_checkpoint_state;
    std::array<T_vec, 2> recovery_checkpoint_tangent;
    std::array<bool, 2> recovery_checkpoint_initialized{{false, false}};
    std::array<bool, 2> recovery_checkpoint_valid{{false, false}};
    std::array<T, 2> recovery_checkpoint_parameter{{T(0), T(0)}};
    std::array<T, 2> recovery_checkpoint_parameter_tangent{{T(0), T(0)}};
    bool last_accepted_checkpoint_valid = false;
    T last_accepted_parameter = T(0);
    T last_accepted_parameter_tangent = T(0);

    void update_last_accepted_checkpoint(
        const T_vec& state,
        const T_vec& tangent,
        const T parameter,
        const T parameter_tangent)
    {
        vec_ops->assign(state, x_last_accepted);
        vec_ops->assign(tangent, x_last_accepted_s);
        last_accepted_parameter = parameter;
        last_accepted_parameter_tangent = parameter_tangent;
        last_accepted_checkpoint_valid = true;
    }

    void capture_recovery_checkpoint(const unsigned int index)
    {
        if(index >= recovery_checkpoint_valid.size())
        {
            return;
        }
        if(!recovery_checkpoint_initialized[index])
        {
            vec_ops->init_vector(recovery_checkpoint_state[index]);
            vec_ops->start_use_vector(recovery_checkpoint_state[index]);
            vec_ops->init_vector(recovery_checkpoint_tangent[index]);
            vec_ops->start_use_vector(recovery_checkpoint_tangent[index]);
            recovery_checkpoint_initialized[index] = true;
        }
        if(!last_accepted_checkpoint_valid)
        {
            return;
        }
        vec_ops->assign(
            x_last_accepted,
            recovery_checkpoint_state[index]);
        vec_ops->assign(
            x_last_accepted_s,
            recovery_checkpoint_tangent[index]);
        recovery_checkpoint_parameter[index] = last_accepted_parameter;
        recovery_checkpoint_parameter_tangent[index] =
            last_accepted_parameter_tangent;
        recovery_checkpoint_valid[index] = true;
    }

    void clear_recovery_checkpoints()
    {
        for(unsigned int index = 0;
            index < recovery_checkpoint_initialized.size();
            ++index)
        {
            if(!recovery_checkpoint_initialized[index])
            {
                continue;
            }
            vec_ops->stop_use_vector(recovery_checkpoint_tangent[index]);
            vec_ops->free_vector(recovery_checkpoint_tangent[index]);
            vec_ops->stop_use_vector(recovery_checkpoint_state[index]);
            vec_ops->free_vector(recovery_checkpoint_state[index]);
            recovery_checkpoint_initialized[index] = false;
            recovery_checkpoint_valid[index] = false;
        }
    }

    void set_pending_endpoint_reason(endpoint_reason_t reason)
    {
        endpoint_state.observe(reason);
    }

    void mark_last_curve_point(endpoint_reason_t reason)
    {
        if(bif_diag != nullptr)
        {
            bif_diag->set_last_endpoint_reason(reason);
        }
        if(container::is_incomplete_endpoint(reason))
        {
            endpoint_state.mark_incomplete();
        }
        current_semicurve_endpoint_reason = reason;
    }

    void add_solution_to_curve(
        const T& lambda,
        const T_vec& x,
        const bool force_store,
        endpoint_reason_t endpoint_reason = endpoint_reason_t::none)
    {
        const bool should_force_store =
            force_store || container::is_terminal_endpoint(endpoint_reason);
        if(container::is_incomplete_endpoint(endpoint_reason))
        {
            endpoint_state.mark_incomplete();
        }
        ++current_semicurve_points;
        current_semicurve_last_parameter = lambda;
        if(endpoint_reason != endpoint_reason_t::none)
        {
            current_semicurve_endpoint_reason = endpoint_reason;
        }
        if(solution_postprocessor)
        {
            vec_ops->assign(x, x_output);
            solution_postprocessor(x_output);
            bif_diag->add(lambda, x_output, should_force_store, endpoint_reason);
        }
        else
        {
            bif_diag->add(lambda, x, should_force_store, endpoint_reason);
        }
    }

private:
    void set_all_vectors()
    {
        
        vec_ops->init_vector(x_check); vec_ops->start_use_vector(x_check);
        vec_ops->init_vector(x_output); vec_ops->start_use_vector(x_output);
        vec_ops->init_vector(x_start); vec_ops->start_use_vector(x_start);
        vec_ops->init_vector(x0); vec_ops->start_use_vector(x0);
        vec_ops->init_vector(x0_s); vec_ops->start_use_vector(x0_s);
        vec_ops->init_vector(x1_s); vec_ops->start_use_vector(x1_s);
        vec_ops->init_vector(x1); vec_ops->start_use_vector(x1);
        vec_ops->init_vector(x1_back); vec_ops->start_use_vector(x1_back);
        vec_ops->init_vector(x_knot_sample); vec_ops->start_use_vector(x_knot_sample);
        vec_ops->init_vector(x_branch_intersection); vec_ops->start_use_vector(x_branch_intersection);
        vec_ops->init_vector(x_pending_branch_event); vec_ops->start_use_vector(x_pending_branch_event);
        vec_ops->init_vector(x_last_accepted); vec_ops->start_use_vector(x_last_accepted);
        vec_ops->init_vector(x_last_accepted_s); vec_ops->start_use_vector(x_last_accepted_s);
    }
    void unset_all_vectors()
    {
        vec_ops->stop_use_vector(x_pending_branch_event); vec_ops->free_vector(x_pending_branch_event);
        vec_ops->stop_use_vector(x_last_accepted_s); vec_ops->free_vector(x_last_accepted_s);
        vec_ops->stop_use_vector(x_last_accepted); vec_ops->free_vector(x_last_accepted);
        vec_ops->stop_use_vector(x_branch_intersection); vec_ops->free_vector(x_branch_intersection);
        vec_ops->stop_use_vector(x_output); vec_ops->free_vector(x_output);
        vec_ops->stop_use_vector(x_check); vec_ops->free_vector(x_check);
        vec_ops->stop_use_vector(x_start); vec_ops->free_vector(x_start);
        vec_ops->stop_use_vector(x0); vec_ops->free_vector(x0);
        vec_ops->stop_use_vector(x0_s); vec_ops->free_vector(x0_s);
        vec_ops->stop_use_vector(x1_s); vec_ops->free_vector(x1_s);
        vec_ops->stop_use_vector(x1); vec_ops->free_vector(x1);
        vec_ops->stop_use_vector(x1_back); vec_ops->free_vector(x1_back);
        vec_ops->stop_use_vector(x_knot_sample); vec_ops->free_vector(x_knot_sample);
    }


    bool get_solution(const T& lambda_fix, T_vec& x_)
    {
        bool converged;
        converged = newton->solve(nonlin_op, x_, lambda_fix);
        return(converged);
    }
    

    bool interpolate_solutions(const T& lambda_star, const T& lambda_0_, const T_vec& x0_,  T& lambda_1_, T_vec& x1_)
    {
        T w = (lambda_star - lambda_0_)/(lambda_1_ - lambda_0_);
        T _w = T(1) - w;
        vec_ops->add_mul(_w, x0_, w, x1_);
        lambda_1_ = lambda_star;
        bool res = get_solution(lambda_star, x1_);
        if(!res)
        {
            log->error("continuation::interpolate_solutions: newton solver for interpolation failed to converge.");
        }
        return(res);
    }

    bool check_vector_distances()
    {

        vec_ops->assign_mul(T(1), x_start, T(-1), x1, x_check);
        T norm_distance_ref = vec_ops->norm_l2(x_start);
        T norm_distance = vec_ops->norm_l2(x_check);
        T epsilon_l = epsilon*((norm_distance_ref>1.0)?norm_distance_ref:1.0);
        if(norm_distance < epsilon_l)
        {
            log->info_f("continuation::check_intersection::check_vector_distances: vectors coincide with distance = %le, distance reference norm = %le and epsilon = %le", norm_distance, norm_distance_ref, epsilon_l);
            return(true);
        }
        else
        {
            log->warning_f("continuation::check_intersection::check_vector_distances: failed with distance = %le, distance reference norm = %le and epsilon = %le", norm_distance, norm_distance_ref, epsilon_l);
            return(false);
        }
    }

    bools2 check_intersection(
        T lambda_star,
        endpoint_reason_t endpoint_reason = endpoint_reason_t::none) //check current interseciton with the parameter value lambda_star
    {
        if( (lambda_star - lambda1)*(lambda_star - lambda0)<=T(0.0) )
        {
            

            bool ret = interpolate_solutions(lambda_star, lambda0, x0, lambda1, x1);
            if(!ret)
            {
                log->warning_f("continuation::check_intersection::interpolate_solutions: returned failed for lambda_star = %le, lambda_0 = %le, lambda_1 = %le", lambda_star, lambda0, lambda1);
                fail_flag = true;
                set_pending_endpoint_reason(endpoint_reason_t::knot_interpolation_failure);
                return(bools2(true, true));
            }
            if(endpoint_reason != endpoint_reason_t::none)
            {
                set_pending_endpoint_reason(endpoint_reason);
                if(endpoint_reason == endpoint_reason_t::boundary_min ||
                   endpoint_reason == endpoint_reason_t::boundary_max)
                {
                    return(bools2(true, false));
                }
            }
            bool vectors_coincide = check_vector_distances();

            return(bools2(true, vectors_coincide));
        }
        else if(lambda1>lambda_max) // if we somehow magically sliped out?!
        {
            if(endpoint_reason == endpoint_reason_t::boundary_max)
            {
                log->warning_f("continuation::check_intersection: passed lambda_max = %le without bracketed interpolation, lambda_1 = %le.", lambda_max, lambda1);
                set_pending_endpoint_reason(endpoint_reason_t::boundary_max);
            }
            else
            {
                log->error_f("continuation::check_intersection: slipped out with lambda_max = %le, lambda_1 = %le.", lambda_max, lambda1);
                fail_flag = true;
                set_pending_endpoint_reason(endpoint_reason_t::hard_failure);
            }
            return(bools2(true, true));            
        }
        else if(lambda1<lambda_min) // if we somehow magically sliped out?!
        {
            if(endpoint_reason == endpoint_reason_t::boundary_min)
            {
                log->warning_f("continuation::check_intersection: passed lambda_min = %le without bracketed interpolation, lambda_1 = %le.", lambda_min, lambda1);
                set_pending_endpoint_reason(endpoint_reason_t::boundary_min);
            }
            else
            {
                log->error_f("continuation::check_intersection: slipped out with lambda_min = %le, lambda_1 = %le.", lambda_min, lambda1);
                fail_flag = true;
                set_pending_endpoint_reason(endpoint_reason_t::hard_failure);
            }
            return(bools2(true, true));
        }
        else
        {
            return(bools2(false, false));
        }
    
    }



    void check_returning()
    {
        vec_ops->assign(x1, x1_back);
        const T lambda1_before_check = lambda1;
        const endpoint_reason_t endpoint_reason_before_check =
            endpoint_state.pending_reason();
        bools2 returned = check_intersection(lambda_start);
        std::string bool1_l = (returned.first?"true":"false");
        std::string bool2_l = (returned.second?"true":"false");

        log->info_f("continuation::check_returning: returned (%s,%s)", bool1_l.c_str(), bool2_l.c_str() );
        
        if(returned.second) //intersecting a starting point
        {
            if(fail_flag)
            {
                break_semicurve++;
                fail_flag = true;
                continue_next_step = false;

            }
            else
            {
                break_semicurve = 2;
                fail_flag = false;
                set_pending_endpoint_reason(endpoint_reason_t::closed_return);
                continue_next_step = false;

            }
        }
        else
        {
            vec_ops->assign(x1_back, x1);
            lambda1 = lambda1_before_check;
            endpoint_state.replace_pending(endpoint_reason_before_check);
        }
    }

    void check_interval()
    {
        bools2 intersect_min(false, false);
        bools2 intersect_max(false, false);

        if(lambda1 < lambda_min)
        {
            intersect_min = check_boundary_intersection(
                lambda_min,
                endpoint_reason_t::boundary_min,
                endpoint_reason_t::boundary_min_approximate);
        }
        else if(lambda1 > lambda_max)
        {
            intersect_max = check_boundary_intersection(
                lambda_max,
                endpoint_reason_t::boundary_max,
                endpoint_reason_t::boundary_max_approximate);
        }
        else
        {
            intersect_min = check_boundary_intersection(
                lambda_min,
                endpoint_reason_t::boundary_min,
                endpoint_reason_t::boundary_min_approximate);
            if(!intersect_min.first)
            {
                intersect_max = check_boundary_intersection(
                    lambda_max,
                    endpoint_reason_t::boundary_max,
                    endpoint_reason_t::boundary_max_approximate);
            }
        }

        if( intersect_min.first || intersect_max.first )
        {
            if(intersect_min.first)
            {
                log->warning_f(
                    "continuation::check_interval: reached lambda_min = %le; stopping semicurve at parameter boundary.",
                    double(lambda_min));
            }
            if(intersect_max.first)
            {
                log->warning_f(
                    "continuation::check_interval: reached lambda_max = %le; stopping semicurve at parameter boundary.",
                    double(lambda_max));
            }
            break_semicurve++;
            fail_flag = false;
            continue_next_step = false;
        }


    }

    bool parameter_is_within_bounds(const T& value) const
    {
        const auto magnitude = [](const T& item) -> T
        {
            using std::abs;
            return abs(item);
        };
        const T scale = std::max<T>(
            T(1),
            std::max<T>(
                std::max<T>(magnitude(lambda_min), magnitude(lambda_max)),
                magnitude(value)));
        const T tolerance = T(64)*std::numeric_limits<T>::epsilon()*scale;
        return value >= lambda_min - tolerance &&
               value <= lambda_max + tolerance;
    }

    bools2 check_boundary_intersection(
        const T lambda_boundary,
        const endpoint_reason_t exact_reason,
        const endpoint_reason_t approximate_reason)
    {
        const bool bracketed =
            (lambda_boundary - lambda1)*
            (lambda_boundary - lambda0) <= T(0);
        const bool already_outside =
            exact_reason == endpoint_reason_t::boundary_max
                ? lambda1 > lambda_boundary
                : lambda1 < lambda_boundary;
        if(!bracketed && !already_outside)
        {
            return bools2(false, false);
        }

        vec_ops->assign(x1, x1_back);
        const T converged_lambda = lambda1;
        if(bracketed &&
           interpolate_solutions(
               lambda_boundary,
               lambda0,
               x0,
               lambda1,
               x1))
        {
            set_pending_endpoint_reason(exact_reason);
            return bools2(true, false);
        }

        if(preserve_last_converged_boundary_point)
        {
            vec_ops->assign(x0, x1);
            lambda1 = lambda0;
            fail_flag = false;
            set_pending_endpoint_reason(approximate_reason);
            log->warning_f(
                "continuation::check_interval: exact boundary refinement at lambda = %le failed; preserving the preceding accepted in-bounds point at lambda = %le and marking an approximate boundary endpoint.",
                double(lambda_boundary),
                double(lambda1));
            return bools2(true, false);
        }

        vec_ops->assign(x1_back, x1);
        lambda1 = converged_lambda;
        fail_flag = true;
        set_pending_endpoint_reason(
            endpoint_reason_t::knot_interpolation_failure);
        log->warning_f(
            "continuation::check_interval: exact boundary refinement at lambda = %le failed and approximate endpoints are disabled.",
            double(lambda_boundary));
        return bools2(true, true);
    }

    bool interpolate_knot_sample(
        const T& lambda_sample,
        const T& lambda_left,
        const T_vec& value_left,
        const T& lambda_right,
        const T_vec& value_right,
        T_vec& sample)
    {
        if(lambda_right == lambda_left)
        {
            return false;
        }

        const T weight =
            (lambda_sample - lambda_left)/
            (lambda_right - lambda_left);
        vec_ops->assign_mul(
            T(1) - weight,
            value_left,
            weight,
            value_right,
            sample);
        const bool converged = get_solution(lambda_sample, sample);
        if(!converged)
        {
            log->warning_f(
                "continuation::interpolate_knot_sample: observational Newton correction failed at lambda = %le inside accepted step [%le, %le].",
                double(lambda_sample),
                double(lambda_left),
                double(lambda_right));
        }
        return converged;
    }

    void sample_intersected_knots(
        bool& force_accepted_point)
    {
        struct knot_candidate
        {
            T requested;
            T attempted;
            T progress;
        };

        if(lambda1 == lambda0)
        {
            return;
        }

        std::vector<knot_candidate> candidates;
        for(const auto& knot: *knots)
        {
            const T requested = knot;
            T attempted = requested;
            if(knot_resolver)
            {
                knot_resolver(requested, attempted);
            }
            if(!parameter_is_bracketed(attempted, lambda0, lambda1))
            {
                continue;
            }
            candidates.push_back(
                knot_candidate{
                    requested,
                    attempted,
                    (attempted - lambda0)/(lambda1 - lambda0)});
        }
        std::sort(
            candidates.begin(),
            candidates.end(),
            [](const knot_candidate& left, const knot_candidate& right)
            {
                return left.progress < right.progress;
            });

        for(const auto& candidate: candidates)
        {
            const auto result = sample_knot_observationally(
                candidate.requested,
                candidate.attempted,
                lambda0,
                x0,
                lambda1,
                x1,
                x_knot_sample,
                [this](
                    const T& parameter,
                    const T& parameter_left,
                    const T_vec& value_left,
                    const T& parameter_right,
                    const T_vec& value_right,
                    T_vec& sample)
                {
                    return interpolate_knot_sample(
                        parameter,
                        parameter_left,
                        value_left,
                        parameter_right,
                        value_right,
                        sample);
                },
                [this](
                    const T& requested,
                    const T& parameter_left,
                    const T_vec& value_left,
                    const T& parameter_right,
                    const T_vec& value_right,
                    T& effective,
                    T_vec& sample)
                {
                    return knot_relocator &&
                           knot_relocator(
                               requested,
                               parameter_left,
                               value_left,
                               parameter_right,
                               value_right,
                               effective,
                               sample);
                });

            if(!result.sampled())
            {
                if(allow_knot_interpolation_failure)
                {
                    log->warning_f(
                        "continuation::start_semicurve: observational sample for requested knot %le failed at effective knot %le; the accepted continuation state is unchanged and sampling is skipped by policy.",
                        double(result.requested_parameter),
                        double(result.attempted_parameter));
                    continue;
                }

                force_accepted_point = true;
                log->warning_f(
                    "continuation::start_semicurve: observational sample for requested knot %le failed at effective knot %le; preserving the curve and force-storing the accepted endpoint at lambda = %le as the nearest available observation.",
                    double(result.requested_parameter),
                    double(result.attempted_parameter),
                    double(lambda1));
                continue;
            }

            add_solution_to_curve(
                result.sampled_parameter,
                x_knot_sample,
                true);
            if(result.relocated())
            {
                log->warning_f(
                    "continuation::start_semicurve: stored an observational sample for requested knot %le after local relocation from effective knot %le to lambda = %le; accepted state, tangent, and chart remain unchanged.",
                    double(result.requested_parameter),
                    double(result.attempted_parameter),
                    double(result.sampled_parameter));
            }
            else
            {
                log->info_f(
                    "continuation::start_semicurve: stored an observational knot sample at lambda = %le without promoting it to the continuation base.",
                    double(result.sampled_parameter));
            }
        }
    }

    void obtain_seed_tangent()
    {
        if(seed_tangent_cache.valid())
        {
            const int cached_direction = seed_tangent_cache.stored_direction();
            seed_tangent_cache.restore(direction, x0_s, lambda0_s);
            const bool same_direction = direction == cached_direction;
            log->info_f(
                "continuation::start_semicurve: reused cached seed tangent: current direction = %i, cached direction = %i, sign flip = %i, lambda_s = %le.",
                direction,
                cached_direction,
                same_direction ? 0 : 1,
                double(lambda0_s));
            if(init_tangent->validate_tangent_candidate(
                   nonlin_op,
                   x0,
                   lambda0,
                   x0_s,
                   lambda0_s,
                   predict->get_initial_ds(),
                   "cached sign-adjusted seed tangent"))
            {
                return;
            }
            log->warning(
                "continuation::start_semicurve: cached sign-adjusted seed tangent failed chart validation; recomputing it.");
            seed_tangent_cache.clear();
        }

        init_tangent->execute(
            nonlin_op,
            T(direction),
            x0,
            lambda0,
            x0_s,
            lambda0_s,
            predict->get_initial_ds());
        seed_tangent_cache.store(x0_s, lambda0_s, direction);
        log->info_f(
            "continuation::start_semicurve: cached seed tangent for direction = %i, lambda_s = %le.",
            direction,
            double(lambda0_s));
    }


    void start_semicurve()
    {
        //assume that x0 and lambda0 are valid solutions, so that ||F(x_0,lambda_0)||<eps
        try
        {
            log->info_f("continuation::start_semicurve: starting semicurve with direction = %i", direction);
            branch_event.clear();
            obtain_seed_tangent();
            update_last_accepted_checkpoint(
                x0,
                x0_s,
                lambda0,
                lambda0_s);
        }
        catch(const std::exception& e)
        {
//            throw std::runtime_error(std::string("continuation::start_semicurve:") + std::string(e.what()) );
            log->error_f("continuation::start_semicurve exception init_tangent: %s\n", e.what());
            break_semicurve++;
            fail_flag = true;
            hard_failure = true;
            endpoint_state.mark_incomplete();
            current_semicurve_endpoint_reason =
                endpoint_reason_t::hard_failure;
            current_semicurve_failure =
                continuation_failure_kind::initial_tangent;
            current_semicurve_failure_message = e.what();
        }
        if(!fail_flag)
        {        
            add_solution_to_curve(lambda0, x0, true); //add initial knot, force save data!
            continuation_step->reset(); //resets all data for initial continuation stepping
            accepted_progress_monitor.reset();
            unsigned int s;
            for(s=0;s<max_S;s++)
            {
                endpoint_state.reset_step();
                try
                {
                    continuation_step->solve(nonlin_op, x0, lambda0, x0_s, lambda0_s, x1, lambda1, x1_s, lambda1_s);
                    bool did_knot_interpolation = false;
                    const bool isotropy_transition_found =
                        continuation_step->has_isotropy_transition();
                    if(isotropy_transition_found)
                    {
                        const auto& event = continuation_step->last_isotropy_transition();
                        log->warning_f(
                            "continuation::start_semicurve: stopping at an isotropy transition C_%lu -> C_%lu at lambda = %le after %u refinements; transverse ratio changed from %le to %le.",
                            static_cast<unsigned long>(event.previous_order),
                            static_cast<unsigned long>(event.candidate_order),
                            double(lambda1),
                            event.refinements,
                            double(event.previous_transverse_ratio),
                            double(event.candidate_transverse_ratio));
                        continue_next_step = false;
                        break_semicurve++;
                        set_pending_endpoint_reason(
                            endpoint_reason_t::symmetry_intersection);
                    }
                    if(continue_next_step)
                    {
                        vec_ops->assign_mul(T(1), x1, T(-1), x0, x_check);
                        const T accepted_progress =
                            vec_ops->norm_rank1(x_check, lambda1-lambda0);
                        if(accepted_progress_monitor.observe(accepted_progress))
                        {
                            log->warning_f(
                                "continuation::start_semicurve: no accepted-state progress over %u steps; accumulated progress = %le. Stopping semicurve.",
                                progress_policy.window_size,
                                double(accepted_progress_monitor.accumulated_progress()));
                            add_solution_to_curve(
                                lambda1,
                                x1,
                                true,
                                endpoint_reason_t::no_progress);
                            update_last_accepted_checkpoint(
                                x1,
                                x1_s,
                                lambda1,
                                lambda1_s);
                            continue_next_step = false;
                            break_semicurve++;
                            endpoint_state.mark_incomplete();
                            current_semicurve_failure =
                                continuation_failure_kind::no_progress;
                            current_semicurve_failure_message =
                                "accepted-state progress watchdog stopped the semicurve";
                            break;
                        }
                    }
                    auto branch_detection = container::branch_intersection_detection::none;
                    if(continue_next_step && branch_intersection_checker)
                    {
                        T hit_lambda = lambda1;
                        int hit_curve_number = -1;
                        uint64_t hit_segment_id = 0;
                        T hit_forward_steps_ahead = T(0);
                        T hit_endpoint_distance_step_ratio = T(0);
                        container::curve_provenance hit_target_provenance;
                        std::string hit_reason;
                        branch_detection = branch_intersection_checker(
                            lambda0,
                            x0,
                            lambda1,
                            x1,
                            hit_lambda,
                            x_branch_intersection,
                            hit_curve_number,
                            hit_segment_id,
                            hit_forward_steps_ahead,
                            hit_endpoint_distance_step_ratio,
                            hit_target_provenance,
                            hit_reason);
                        if(branch_detection != container::branch_intersection_detection::none &&
                           !parameter_is_within_bounds(hit_lambda))
                        {
                            branch_detection =
                                container::branch_intersection_detection::none;
                            branch_event.clear();
                        }
                        if(branch_detection == container::branch_intersection_detection::verified)
                        {
                            const T corrected_hit_lambda = hit_lambda;
                            if(branch_event.matches(
                                   hit_curve_number,
                                   hit_segment_id))
                            {
                                hit_target_provenance =
                                    branch_event.target_provenance();
                                hit_lambda = branch_event.lambda();
                                vec_ops->assign(
                                    x_pending_branch_event,
                                    x_branch_intersection);
                                std::ostringstream verified_reason;
                                verified_reason
                                    << "verified branch encounter with curve "
                                    << hit_curve_number
                                    << " near predicted lambda = " << hit_lambda
                                    << " after a corrected state reached that branch at lambda = "
                                    << corrected_hit_lambda;
                                hit_reason = verified_reason.str();
                            }
                            branch_event.clear();
                            lambda1 = hit_lambda;
                            vec_ops->assign(x_branch_intersection, x1);
                            did_knot_interpolation = true;
                            continue_next_step = false;
                            break_semicurve++;
                            set_pending_endpoint_reason(
                                hit_target_provenance.is_analytical()
                                    ? endpoint_reason_t::analytical_branch
                                    : endpoint_reason_t::known_branch);
                            log->warning_f(
                                "continuation::start_semicurve: stopped semicurve at lambda = %le due to %s.",
                                double(lambda1),
                                hit_reason.empty() ? "known branch intersection" : hit_reason.c_str());
                        }
                        else if(branch_detection == container::branch_intersection_detection::forward_approach)
                        {
                            branch_event.update(
                                hit_lambda,
                                hit_curve_number,
                                hit_segment_id,
                                hit_target_provenance);
                            vec_ops->assign(
                                x_branch_intersection,
                                x_pending_branch_event);
                            branch_event.increment_refinements();
                            const bool generic_event_localized =
                                container::forward_branch_event_is_localized(
                                    branch_event.refinements(),
                                    hit_forward_steps_ahead,
                                    hit_endpoint_distance_step_ratio,
                                    minimum_branch_refinements_for_verification,
                                    maximum_verified_branch_steps_ahead,
                                    maximum_verified_branch_distance_step_ratio);
                            const bool analytical_event_localized =
                                localize_analytical_branch_targets &&
                                branch_event.target_provenance().is_analytical() &&
                                branch_event.refinements() >=
                                    minimum_branch_refinements_for_verification &&
                                branch_event.prediction_is_stable(
                                    analytical_branch_parameter_tolerance);
                            const bool event_localized =
                                generic_event_localized ||
                                analytical_event_localized;
                            if(event_localized)
                            {
                                lambda1 = branch_event.lambda();
                                vec_ops->assign(x_pending_branch_event, x1);
                                did_knot_interpolation = true;
                                continue_next_step = false;
                                break_semicurve++;
                                set_pending_endpoint_reason(
                                    branch_event.target_provenance().is_analytical()
                                        ? endpoint_reason_t::analytical_branch
                                        : endpoint_reason_t::known_branch);
                                log->warning_f(
                                    "continuation::start_semicurve: localized branch encounter with curve %i at lambda = %le after %u refinements (estimated steps ahead = %le, distance/step = %le); stopping before the singular corrector can switch branches.",
                                    hit_curve_number,
                                    double(lambda1),
                                    branch_event.refinements(),
                                    double(hit_forward_steps_ahead),
                                    double(hit_endpoint_distance_step_ratio));
                                branch_event.clear();
                            }
                            auto reduction = step_retry_result::retry;
                            if(continue_next_step &&
                               branch_event.refinements() < maximum_branch_refinements)
                            {
                                reduction = continuation_step->reduce_next_step(
                                    branch_refinement_step_factor);
                            }
                            if(continue_next_step &&
                               (branch_event.refinements() >= maximum_branch_refinements ||
                               reduction != step_retry_result::retry)
                              )
                            {
                                continue_next_step = false;
                                break_semicurve++;
                                endpoint_state.mark_incomplete();
                                set_pending_endpoint_reason(
                                    endpoint_reason_t::unresolved_branch_intersection);
                                current_semicurve_failure =
                                    continuation_failure_kind::unresolved_intersection;
                                current_semicurve_failure_message =
                                    "branch intersection could not be verified";
                                log->warning_f(
                                    "continuation::start_semicurve: could not verify a predicted branch intersection after %u refinements; stopping at lambda = %le without inserting a known-branch state.",
                                    branch_event.refinements(),
                                    double(lambda1));
                            }
                            else if(continue_next_step)
                            {
                                log->warning_f(
                                    "continuation::start_semicurve: possible branch encounter near lambda = %le is not yet verified; refinement %u of %u will continue with a smaller dS. %s",
                                    double(hit_lambda),
                                    branch_event.refinements(),
                                    maximum_branch_refinements,
                                    hit_reason.c_str());
                            }
                        }
                        else
                        {
                            if(branch_event.active())
                            {
                                branch_event.note_miss(2);
                            }
                        }
                    }
                    bool self_intersection_found = false;
                    if(continue_next_step && self_intersection_checker)
                    {
                        T hit_lambda = lambda1;
                        std::string hit_reason;
                        self_intersection_found = self_intersection_checker(
                            bif_diag,
                            lambda0,
                            x0,
                            lambda1,
                            x1,
                            hit_lambda,
                            x_branch_intersection,
                            hit_reason);
                        if(self_intersection_found)
                        {
                            lambda1 = hit_lambda;
                            vec_ops->assign(x_branch_intersection, x1);
                            did_knot_interpolation = true;
                            continue_next_step = false;
                            break_semicurve++;
                            set_pending_endpoint_reason(endpoint_reason_t::self_intersection);
                            log->warning_f(
                                "continuation::start_semicurve: stopped semicurve at lambda = %le due to %s.",
                                double(lambda1),
                                hit_reason.empty() ? "curve-local self intersection" : hit_reason.c_str());
                        }
                    }
                    if(continue_next_step &&
                       (lambda1 < lambda_min || lambda1 > lambda_max))
                    {
                        // A verified branch encounter takes precedence over a
                        // boundary crossed by the same accepted step.
                        check_interval();
                    }
                    if((s>1)&&(!just_interpolated))
                    {
                        if(continue_next_step)
                        {
                            check_interval();
                        }
                        if(continue_next_step && !branch_event.active())
                        {
                            check_returning();
                        }
                        if(continue_next_step && !branch_event.active())
                        {
                            sample_intersected_knots(
                                did_knot_interpolation);
                        }
                    }
                    else
                    {
                        just_interpolated = false;
                    }
                    //if try blocks passes, THIS is executed:
                    update_last_accepted_checkpoint(
                        x1,
                        x1_s,
                        lambda1,
                        lambda1_s);
                    add_solution_to_curve(
                        lambda1,
                        x1,
                        did_knot_interpolation,
                        endpoint_state.pending_reason());
                    if(isotropy_transition_found)
                    {
                        detail::record_symmetry_intersection_if_available(
                            bif_diag,
                            continuation_step->last_isotropy_transition(),
                            0);
                    }
                    endpoint_state.reset_step();
                    
                    vec_ops->assign(x1, x0);
                    vec_ops->assign(x1_s, x0_s);
                    lambda0 = lambda1;
                    lambda0_s = lambda1_s;
                }
                catch(const std::exception& e)
                {
                    log->error_f("continuation::start_semicurve exception continuation_step: %s\n", e.what());
                    break_semicurve++;
                    fail_flag = true; 
                    hard_failure = true;
                    endpoint_state.mark_incomplete();
                    mark_last_curve_point(endpoint_reason_t::hard_failure);
                    const auto& attempt =
                        continuation_step->last_attempt_state();
                    current_semicurve_failure =
                        attempt.failure == continuation_failure_kind::none
                            ? continuation_failure_kind::unknown
                            : attempt.failure;
                    current_semicurve_failure_message = e.what();
                    current_semicurve_attempted_step =
                        attempt.attempted_step;
                    current_semicurve_retry_count = attempt.retry_count;
                    continue_next_step = false;                   
                }
                if(!continue_next_step)
                { 
                   break;
                }

            }
            if(s==max_S)
            {
                log->warning_f("continuation::start_semicurve: reached maximum steps = %i", s);
                continue_next_step = false;
                break_semicurve++;
                endpoint_state.mark_incomplete();
                mark_last_curve_point(endpoint_reason_t::max_steps);
                current_semicurve_failure =
                    continuation_failure_kind::maximum_steps;
                current_semicurve_failure_message =
                    "maximum continuation steps reached";
            }

        }       

    }



};

}




#endif // CONTINUATION_HPP
