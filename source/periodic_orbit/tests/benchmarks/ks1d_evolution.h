#ifndef PERIODIC_ORBIT_TESTS_KS1D_EVOLUTION_H
#define PERIODIC_ORBIT_TESTS_KS1D_EVOLUTION_H

#include <nonlinear_operators/Kuramoto_Sivashinskiy_1D/kuramoto_sivashinskiy_1d.h>

namespace ks1d_orbit
{
// Physical time evolves by -F. Views of a serial replay workspace bind their
// parameter by value; trial and replay workspaces must be constructed separately.
template <class Space, class FFT>
class evolution
{
    using T       = typename Space::scalar_type;
    using X       = typename Space::vector_type;
    using ks_type = nonlinear_operators::kuramoto_sivashinskiy_1d<Space, FFT>;
    std::shared_ptr<ks_type> ks_;
    T                        lambda_;

public:
    evolution( const std::shared_ptr<Space> &space, std::size_t grid, T a, T b, T lambda )
        : ks_( std::make_shared<ks_type>( a, b, grid, space ) ), lambda_( lambda )
    {
    }
    void set_parameter( T lambda )
    {
        lambda_ = lambda;
    }
    evolution with_parameter( T lambda ) const
    {
        evolution view = *this;
        view.set_parameter( lambda );
        return view;
    }

    bool evaluate_rate( const X &u, X &out ) const
    {
        ks_->F( u, lambda_, out );
        ks_->get_vec_ops_ref()->scale( T( -1 ), out );
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
        ks_->set_linearization_point( u, lambda_ );
        ks_->jacobian_u( v, out );
        ks_->get_vec_ops_ref()->scale( T( -1 ), out );
        return true;
    }
    bool parameter_derivative( T, const X &u, X &out ) const
    {
        ks_->jacobian_alpha( u, lambda_, out );
        ks_->get_vec_ops_ref()->scale( T( -1 ), out );
        return true;
    }
};
}
#endif
