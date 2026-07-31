#ifndef __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_SCALED_REAL_OPERATOR_H__
#define __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_SCALED_REAL_OPERATOR_H__

#include <type_traits>

namespace stability
{
namespace eigensolvers
{
namespace transformations
{

template<class VectorOperations, class RealOperator>
class scaled_real_operator
{
public:
    using vector_operations_type = VectorOperations;
    using operator_type = RealOperator;
    using scalar_type =
        typename vector_operations_type::scalar_type;
    using vector_type =
        typename vector_operations_type::vector_type;

    scaled_real_operator(
        vector_operations_type& vector_operations,
        const operator_type& real_operator,
        scalar_type scale)
        : vector_operations_(vector_operations),
          real_operator_(real_operator),
          scale_(scale)
    {
    }

    bool apply(
        const vector_type& source,
        vector_type& destination) const
    {
        using result_type = decltype(
            real_operator_.apply(source, destination));
        if constexpr(std::is_void<result_type>::value)
        {
            real_operator_.apply(source, destination);
        }
        else if(!static_cast<bool>(
                    real_operator_.apply(source, destination)))
        {
            return false;
        }
        vector_operations_.scale(scale_, destination);
        return
            vector_operations_.check_is_valid_number(destination);
    }

    scalar_type scale() const
    {
        return scale_;
    }

    const operator_type& real_operator() const
    {
        return real_operator_;
    }

private:
    vector_operations_type& vector_operations_;
    const operator_type& real_operator_;
    scalar_type scale_;
};

} // namespace transformations
} // namespace eigensolvers
} // namespace stability

#endif
