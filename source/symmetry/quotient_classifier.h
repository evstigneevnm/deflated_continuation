#ifndef __SYMMETRY_QUOTIENT_CLASSIFIER_H__
#define __SYMMETRY_QUOTIENT_CLASSIFIER_H__

namespace symmetry
{

enum class quotient_solution_kind
{
    not_converged,
    stationary,
    relative_equilibrium
};

inline const char *quotient_solution_kind_name( const quotient_solution_kind kind )
{
    switch ( kind )
    {
    case quotient_solution_kind::not_converged:
        return "not_converged";
    case quotient_solution_kind::stationary:
        return "stationary";
    case quotient_solution_kind::relative_equilibrium:
        return "relative_equilibrium";
    }
    return "unknown";
}

template <class Real>
struct quotient_residual_report
{
    Real projected_residual_norm = Real{};
    Real full_residual_norm = Real{};
    Real projected_tolerance = Real{};
    Real full_tolerance = Real{};
    quotient_solution_kind kind = quotient_solution_kind::not_converged;

    bool quotient_converged() const
    {
        return kind == quotient_solution_kind::stationary || kind == quotient_solution_kind::relative_equilibrium;
    }

    bool stationary() const
    {
        return kind == quotient_solution_kind::stationary;
    }

    bool relative_equilibrium() const
    {
        return kind == quotient_solution_kind::relative_equilibrium;
    }
};

template <class Real>
quotient_residual_report<Real> classify_quotient_residual(
    const Real projected_residual_norm, const Real full_residual_norm, const Real projected_tolerance,
    const Real full_tolerance
)
{
    quotient_residual_report<Real> report;
    report.projected_residual_norm = projected_residual_norm;
    report.full_residual_norm = full_residual_norm;
    report.projected_tolerance = projected_tolerance;
    report.full_tolerance = full_tolerance;

    if ( projected_residual_norm > projected_tolerance )
    {
        report.kind = quotient_solution_kind::not_converged;
    }
    else if ( full_residual_norm <= full_tolerance )
    {
        report.kind = quotient_solution_kind::stationary;
    }
    else
    {
        report.kind = quotient_solution_kind::relative_equilibrium;
    }
    return report;
}

} // namespace symmetry

#endif
