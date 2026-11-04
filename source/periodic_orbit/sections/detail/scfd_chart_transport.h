#ifndef PERIODIC_ORBIT_SCFD_CHART_TRANSPORT_H
#define PERIODIC_ORBIT_SCFD_CHART_TRANSPORT_H

#include <type_traits>
#include <scfd/utils/device_tag.h>

namespace periodic_orbit
{
namespace detail
{
template <class T, class Ordinal>
struct chart_reflector_kernel
{
    const T            *normal;
    T                  *reflector;
    __DEVICE_TAG__ void operator()( Ordinal i ) const
    {
        // Adding the same sign avoids cancellation even near either coordinate axis.
        reflector[i] = normal[i] + ( i == 0 ? ( normal[0] < T( 0 ) ? T( -1 ) : T( 1 ) ) : T( 0 ) );
    }
};

template <class T, class Ordinal>
struct chart_embed_kernel
{
    const T            *coordinates;
    T                  *state;
    __DEVICE_TAG__ void operator()( Ordinal i ) const
    {
        state[i] = i == 0 ? T( 0 ) : coordinates[i - 1];
    }
};

template <class T, class Ordinal>
struct chart_restrict_kernel
{
    const T            *state;
    T                  *coordinates;
    __DEVICE_TAG__ void operator()( Ordinal i ) const
    {
        coordinates[i] = state[i + 1];
    }
};
}

// Flat, local SCFD arrays only. Other layouts can inject a transport with these
// three operations; no host-view fallback is supplied for unsupported layouts.
template <class Space, class CoordinateSpace>
struct scfd_chart_transport
{
    using T            = typename Space::scalar_type;
    using X            = typename Space::vector_type;
    using Z            = typename CoordinateSpace::vector_type;
    using ordinal_type = typename Space::ordinal_type;
    static_assert(
        std::is_same_v<typename Space::memory_type, typename CoordinateSpace::memory_type>,
        "Chart buffers must use the same backend memory"
    );

    void make_reflector( const X &normal, X &out ) const
    {
        typename Space::for_each_type{}(
            detail::chart_reflector_kernel<T, ordinal_type>{ normal.raw_ptr(), out.raw_ptr() }, out.size()
        );
    }
    void embed( const Z &in, X &out ) const
    {
        typename Space::for_each_type{}(
            detail::chart_embed_kernel<T, ordinal_type>{ in.raw_ptr(), out.raw_ptr() }, out.size()
        );
    }
    void restrict( const X &in, Z &out ) const
    {
        if ( out.size() )
            typename Space::for_each_type{}(
                detail::chart_restrict_kernel<T, ordinal_type>{ in.raw_ptr(), out.raw_ptr() }, out.size()
            );
    }
};
}
#endif
