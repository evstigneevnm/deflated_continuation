#ifndef __NUMERICAL_ALGOS_BYPASSABLE_PRECONDITIONER_H__
#define __NUMERICAL_ALGOS_BYPASSABLE_PRECONDITIONER_H__

#include <type_traits>

namespace numerical_algos
{

// Shared through the preconditioner's control block; no extra vector or allocation.
struct preconditioner_bypass
{
    bool enabled = false;
};

template <class Preconditioner>
class bypassable_preconditioner : public Preconditioner, public preconditioner_bypass
{
public:
    using vector_type = typename Preconditioner::vector_type;
    using Preconditioner::apply;
    using Preconditioner::Preconditioner;

    bool apply( vector_type &in_out ) const
    {
        if ( enabled )
            return true;
        if constexpr ( std::is_void_v<decltype( Preconditioner::apply( in_out ) )> )
        {
            Preconditioner::apply( in_out );
            return true;
        }
        else
            return Preconditioner::apply( in_out );
    }
};

} // namespace numerical_algos

#endif
