#ifndef __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_SCALED_REAL_OPERATOR_H__
#define __STABILITY_EIGENSOLVERS_TRANSFORMATIONS_SCALED_REAL_OPERATOR_H__

#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace stability
{
namespace eigensolvers
{
namespace transformations
{

template <class VectorOperations, class RealOperator>
class scaled_real_operator
{
public:
    using vector_operations_type = VectorOperations;
    using operator_type          = RealOperator;
    using scalar_type            = typename vector_operations_type::scalar_type;
    using vector_type            = typename vector_operations_type::vector_type;

    scaled_real_operator(
        std::shared_ptr<vector_operations_type> vector_operations, std::shared_ptr<const operator_type> real_operator,
        scalar_type scale
    )
        : scaled_real_operator(
              vector_operations ? *vector_operations
                                : throw std::invalid_argument( "scaled real operator requires vector operations" ),
              real_operator ? *real_operator
                            : throw std::invalid_argument( "scaled real operator requires an operator" ),
              scale
          )
    {
        vector_operations_owner_ = std::move( vector_operations );
        real_operator_owner_     = std::move( real_operator );
    }

    scaled_real_operator(
        vector_operations_type &vector_operations, const operator_type &real_operator, scalar_type scale
    )
        : vector_operations_( vector_operations ), real_operator_( real_operator ), scale_( scale )
    {
    }

    bool apply( const vector_type &source, vector_type &destination ) const
    {
        using result_type = decltype( real_operator_.apply( source, destination ) );
        if constexpr ( std::is_void<result_type>::value )
        {
            real_operator_.apply( source, destination );
        }
        else if ( !static_cast<bool>( real_operator_.apply( source, destination ) ) )
        {
            return false;
        }
        vector_operations_.scale( scale_, destination );
        return vector_operations_.is_valid_number( destination );
    }

    scalar_type scale() const
    {
        return scale_;
    }

    const operator_type &real_operator() const
    {
        return real_operator_;
    }

private:
    std::shared_ptr<vector_operations_type> vector_operations_owner_;
    std::shared_ptr<const operator_type>    real_operator_owner_;
    vector_operations_type                 &vector_operations_;
    const operator_type                    &real_operator_;
    scalar_type                             scale_;
};

} // namespace transformations
} // namespace eigensolvers
} // namespace stability

#endif
