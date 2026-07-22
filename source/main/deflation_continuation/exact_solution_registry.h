#ifndef __MAIN_DEFLATION_CONTINUATION_EXACT_SOLUTION_REGISTRY_H__
#define __MAIN_DEFLATION_CONTINUATION_EXACT_SOLUTION_REGISTRY_H__

#include <cstddef>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace main_classes
{
namespace deflation_continuation_detail
{

namespace exact_solution_detail
{

template<class...>
using void_t = void;

template<class Operator, class Scalar, class Vector, class = void>
struct has_legacy_solution : std::false_type
{
};

template<class Operator, class Scalar, class Vector>
struct has_legacy_solution<
    Operator,
    Scalar,
    Vector,
    void_t<decltype(std::declval<Operator&>().exact_solution(
        std::declval<const Scalar&>(),
        std::declval<Vector&>()))>> : std::true_type
{
};

template<class Operator, class Scalar, class Vector, class = void>
struct has_indexed_solution : std::false_type
{
};

template<class Operator, class Scalar, class Vector>
struct has_indexed_solution<
    Operator,
    Scalar,
    Vector,
    void_t<decltype(std::declval<Operator&>().exact_solution(
        std::declval<std::size_t>(),
        std::declval<const Scalar&>(),
        std::declval<Vector&>()))>> : std::true_type
{
};

template<class Operator, class = void>
struct has_solution_count : std::false_type
{
};

template<class Operator>
struct has_solution_count<
    Operator,
    void_t<decltype(std::declval<Operator&>().exact_solution_count())>>
    : std::true_type
{
};

template<class Operator, class = void>
struct has_solution_name : std::false_type
{
};

template<class Operator>
struct has_solution_name<
    Operator,
    void_t<decltype(std::declval<Operator&>().exact_solution_name(
        std::declval<std::size_t>()))>> : std::true_type
{
};

} // namespace exact_solution_detail

template<class NonlinearOperator, class Scalar, class Vector>
class exact_solution_registry
{
public:
    explicit exact_solution_registry(NonlinearOperator* nonlinear_operator):
        nonlinear_operator_(nonlinear_operator)
    {
    }

    std::size_t count() const
    {
        if constexpr(exact_solution_detail::has_solution_count<NonlinearOperator>::value)
        {
            return static_cast<std::size_t>(nonlinear_operator_->exact_solution_count());
        }
        else if constexpr(
            exact_solution_detail::has_legacy_solution<
                NonlinearOperator,
                Scalar,
                Vector>::value)
        {
            return 1;
        }
        else
        {
            return 0;
        }
    }

    bool evaluate(
        const std::size_t branch_id,
        const Scalar& parameter,
        Vector& value) const
    {
        if(branch_id >= count())
        {
            return false;
        }

        if constexpr(exact_solution_detail::has_indexed_solution<
                         NonlinearOperator,
                         Scalar,
                         Vector>::value)
        {
            return static_cast<bool>(
                nonlinear_operator_->exact_solution(branch_id, parameter, value));
        }
        else if constexpr(exact_solution_detail::has_legacy_solution<
                              NonlinearOperator,
                              Scalar,
                              Vector>::value)
        {
            if(branch_id != 0)
            {
                return false;
            }
            nonlinear_operator_->exact_solution(parameter, value);
            return true;
        }
        else
        {
            return false;
        }
    }

    std::string name(const std::size_t branch_id) const
    {
        if constexpr(exact_solution_detail::has_solution_name<NonlinearOperator>::value)
        {
            return nonlinear_operator_->exact_solution_name(branch_id);
        }
        else
        {
            std::ostringstream stream;
            stream << "exact_solution_" << branch_id;
            return stream.str();
        }
    }

    template<class Integer, class InvalidBranch>
    std::vector<std::size_t> select(
        const std::vector<Integer>& requested,
        InvalidBranch&& invalid_branch) const
    {
        const std::size_t branch_count = count();
        std::vector<std::size_t> result;
        if(requested.empty())
        {
            result.reserve(branch_count);
            for(std::size_t branch_id = 0; branch_id < branch_count; ++branch_id)
            {
                result.push_back(branch_id);
            }
            return result;
        }

        result.reserve(requested.size());
        for(const auto requested_id: requested)
        {
            const std::size_t branch_id = static_cast<std::size_t>(requested_id);
            if(branch_id >= branch_count)
            {
                invalid_branch(requested_id, branch_count);
                continue;
            }
            result.push_back(branch_id);
        }
        return result;
    }

private:
    NonlinearOperator* nonlinear_operator_;
};

} // namespace deflation_continuation_detail
} // namespace main_classes

#endif // __MAIN_DEFLATION_CONTINUATION_EXACT_SOLUTION_REGISTRY_H__
