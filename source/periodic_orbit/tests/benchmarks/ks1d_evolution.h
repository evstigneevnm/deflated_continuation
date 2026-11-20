#ifndef PERIODIC_ORBIT_TESTS_KS1D_EVOLUTION_H
#define PERIODIC_ORBIT_TESTS_KS1D_EVOLUTION_H

#include <nonlinear_operators/Kuramoto_Sivashinskiy_1D/kuramoto_sivashinskiy_1d.h>

namespace ks1d_orbit
{
// The stationary operator is F; physical time evolves by -F. Each provider
// owns its FFT workspace so a trial flight cannot overwrite replay data.
template <class Space, class FFT>
class evolution
{
    using T = typename Space::scalar_type;
    using X = typename Space::vector_type;
    mutable nonlinear_operators::kuramoto_sivashinskiy_1d<Space, FFT> ks_;
    const T                                                           lambda_;

public:
    evolution( const std::shared_ptr<Space> &space, std::size_t grid, T a, T b, T lambda )
        : ks_( a, b, grid, space ), lambda_( lambda )
    {
    }

    bool evaluate_rate( const X &u, X &out ) const
    {
        ks_.F( u, lambda_, out );
        ks_.get_vec_ops_ref()->scale( T( -1 ), out );
        return true;
    }
    bool evaluate( T, const X &u, X &out ) const
    {
        return evaluate_rate( u, out );
    }

    bool apply_jacobian( T, const X &u, const X &v, X &out ) const
    {
        // F and jacobian_alpha also use physical_u/ux: bind the stage here,
        // never infer a cached linearization from a recycled vector address.
        ks_.set_linearization_point( u, lambda_ );
        ks_.jacobian_u( v, out );
        ks_.get_vec_ops_ref()->scale( T( -1 ), out );
        return true;
    }
    bool parameter_derivative( T, const X &u, X &out ) const
    {
        ks_.jacobian_alpha( u, lambda_, out );
        ks_.get_vec_ops_ref()->scale( T( -1 ), out );
        return true;
    }
};
}
#endif
