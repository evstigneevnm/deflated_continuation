#ifndef __CONTINUATION_INITIAL_TANGENT_CHART_VALIDATOR_H__
#define __CONTINUATION_INITIAL_TANGENT_CHART_VALIDATOR_H__

#include <common/scalar_math.h>
#include <continuation/chart_helpers.h>
#include <continuation/initial_tangent_candidates.h>
#include <continuation/predictor_chart_probe.h>

namespace continuation
{

template<class VectorOperations, class Log, class NonlinearOperator>
class initial_tangent_chart_validator
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;
    using quality_type = tangent_candidate_quality<scalar_type>;

    initial_tangent_chart_validator(VectorOperations* vec_ops_, Log* log_):
        vec_ops(vec_ops_),
        log(log_)
    {
        if constexpr(chart::has_continuation_chart<
                         NonlinearOperator,
                         vector_type,
                         scalar_type>::value)
        {
            vec_ops->init_vector(raw_predictor);
            vec_ops->start_use_vector(raw_predictor);
            vec_ops->init_vector(charted_predictor);
            vec_ops->start_use_vector(charted_predictor);
            vec_ops->init_vector(work);
            vec_ops->start_use_vector(work);
        }
    }

    ~initial_tangent_chart_validator()
    {
        if constexpr(chart::has_continuation_chart<
                         NonlinearOperator,
                         vector_type,
                         scalar_type>::value)
        {
            vec_ops->stop_use_vector(work);
            vec_ops->free_vector(work);
            vec_ops->stop_use_vector(charted_predictor);
            vec_ops->free_vector(charted_predictor);
            vec_ops->stop_use_vector(raw_predictor);
            vec_ops->free_vector(raw_predictor);
        }
    }

    initial_tangent_chart_validator(const initial_tangent_chart_validator&) = delete;
    initial_tangent_chart_validator& operator=(const initial_tangent_chart_validator&) = delete;

    void set_policy(const predictor_chart_policy<scalar_type>& policy_)
    {
        validate_predictor_chart_policy(policy_);
        policy = policy_;
    }

    bool accepts(
        NonlinearOperator* nonlin_op,
        const vector_type& x,
        const scalar_type& lambda,
        const vector_type& x_s,
        const scalar_type& lambda_s,
        const scalar_type& predictor_ds,
        const char* method,
        quality_type* quality)
    {
        if constexpr(!chart::has_continuation_chart<
                          NonlinearOperator,
                          vector_type,
                          scalar_type>::value)
        {
            (void)nonlin_op;
            (void)x;
            (void)lambda;
            (void)x_s;
            (void)lambda_s;
            (void)predictor_ds;
            (void)method;
            (void)quality;
            return true;
        }
        else
        {
            const auto probe = probe_predictor_chart(
                vec_ops,
                log,
                nonlin_op,
                x,
                lambda,
                x_s,
                lambda_s,
                predictor_ds,
                raw_predictor,
                charted_predictor,
                work,
                policy);
            if(quality != nullptr)
            {
                quality->chart_progress_ratio = probe.validation.progress_ratio;
                quality->chart_displacement_ratio = probe.validation.displacement_ratio;
                quality->score +=
                    scalar_type(0.05)*common::scalar_math::abs(
                        probe.validation.progress_ratio - scalar_type(1)) +
                    scalar_type(0.01)*probe.validation.displacement_ratio;
            }

            log->info_f(
                "continuation::initial_tangent: chart probe: method = %s, ds = %le, raw progress = %le, charted progress = %le, progress ratio = %le, chart displacement = %le, displacement ratio = %le, decision = %s.",
                method,
                double(predictor_ds),
                double(probe.raw_tangent_progress),
                double(probe.charted_tangent_progress),
                double(probe.validation.progress_ratio),
                double(probe.chart_displacement),
                double(probe.validation.displacement_ratio),
                predictor_chart_decision_name(probe.validation.decision));
            chart::log_continuation_chart(
                log,
                nonlin_op,
                "continuation::initial_tangent::chart_probe");

            chart::restore_continuation_chart(
                vec_ops,
                log,
                nonlin_op,
                x,
                lambda,
                x_s,
                lambda_s);

            if(probe.validation.rejected())
            {
                log->warning_f(
                    "continuation::initial_tangent: rejected tangent candidate after chart probe: method = %s, decision = %s, raw progress = %le, charted progress = %le, progress ratio = %le, displacement ratio = %le.",
                    method,
                    predictor_chart_decision_name(probe.validation.decision),
                    double(probe.raw_tangent_progress),
                    double(probe.charted_tangent_progress),
                    double(probe.validation.progress_ratio),
                    double(probe.validation.displacement_ratio));
                return false;
            }
            return true;
        }
    }

private:
    VectorOperations* vec_ops;
    Log* log;
    predictor_chart_policy<scalar_type> policy;
    vector_type raw_predictor;
    vector_type charted_predictor;
    vector_type work;
};

} // namespace continuation

#endif // __CONTINUATION_INITIAL_TANGENT_CHART_VALIDATOR_H__
