#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <nmfd/detail/vector_wrap.h>
#include <nmfd/operations/scfd_vector_space.h>
#include <scfd/backend/serial_cpu.h>
#include <symmetry/fourier/residual_translation_orbit_aligner_2d.h>

namespace
{

struct vector_operations : nmfd::operations::scfd_vector_space<scfd::backend::serial_cpu<std::ptrdiff_t>, double>
{
    using base_type = nmfd::operations::scfd_vector_space<scfd::backend::serial_cpu<std::ptrdiff_t>, double>;
    using base_type::base_type;

    // Alignment must not use full-vector host access, even on the serial backend.
    void get( const vector_type &, double *, std::size_t ) const = delete;
    void set( const double *, vector_type &, std::size_t ) const = delete;
};

class test_vector : public nmfd::detail::vector_wrap<vector_operations, false, false>
{
public:
    explicit test_vector( const vector_operations &operations, std::vector<double> values = {} )
        : vector_wrap( operations )
    {
        init();
        start_use();
        operations.assign_scalar( 0., **this );
        if ( !values.empty() )
            operations.base_type::set( values.data(), **this, values.size() );
    }
    operator vector_type &()
    {
        return **this;
    }
    operator const vector_type &() const
    {
        return **this;
    }
    double &operator[]( std::size_t i )
    {
        return ( **this ).raw_ptr()[i];
    }
    double operator[]( std::size_t i ) const
    {
        return ( **this ).raw_ptr()[i];
    }
    std::size_t size() const
    {
        return space_.get_size( **this );
    }
};

void require( const bool condition, const std::string &message )
{
    if ( !condition )
    {
        throw std::runtime_error( message );
    }
}

double distance( const test_vector &left, const test_vector &right )
{
    double result = 0.0;
    for ( std::size_t index = 0; index < left.size(); ++index )
    {
        const double difference = left[index] - right[index];
        result += difference * difference;
    }
    return std::sqrt( result );
}

} // namespace

int main()
{
    try
    {
        vector_operations operations( 3 );
        using aligner_type = symmetry::fourier::residual_translation_orbit_aligner_2d<vector_operations>;
        const std::vector<symmetry::fourier::mode_index<2>> modes{ { 2, 0 }, { 4, 0 }, { 6, 0 } };
        aligner_type                                        aligner( &operations, modes );

        const test_vector reference( operations, { 2.0, 1.0, 0.5 } );
        const test_vector translated( operations, { -2.0, 1.0, -0.5 } );
        const test_vector translated_tangent( operations, { -0.3, -0.2, -0.1 } );
        const test_vector expected_tangent( operations, { 0.3, -0.2, 0.1 } );
        test_vector       aligned( operations ), aligned_tangent( operations );

        aligner.align_orbit_closest_to_reference( reference, translated, aligned );
        require( distance( aligned, reference ) < 1.0e-14, "residual translation aligns the state" );
        aligner.apply_last_alignment( translated_tangent, aligned_tangent );
        require(
            distance( aligned_tangent, expected_tangent ) < 1.0e-14,
            "the selected residual character also aligns the tangent"
        );

        // A quarter shift preserves the even-mode branch; weak odd-mode noise must not hide that orbit.
        vector_operations noisy_operations( 16 );
        auto              noisy_modes = modes;
        noisy_modes.resize( 16, { 1, 0 } );
        aligner_type::policy_type policy;
        policy.absolute_active_tolerance = 1.0e-8;
        aligner_type noisy_aligner( &noisy_operations, noisy_modes, policy );
        test_vector  clean( noisy_operations ), noisy( noisy_operations ), noisy_aligned( noisy_operations );
        clean[0] = 2.0;
        clean[1] = 1.0;
        noisy[0] = -2.0;
        noisy[1] = 1.0;
        noisy[3] = 1.5e-8;
        noisy_aligner.align_orbit_closest_to_reference( clean, noisy, noisy_aligned );
        const double forward = distance( clean, noisy_aligned );
        noisy_aligner.align_orbit_closest_to_reference( noisy, clean, noisy_aligned );
        const double reverse = distance( noisy, noisy_aligned );
        require( std::abs( forward - 1.5e-8 ) < 1.0e-14, "weak source noise hides a quarter-shift duplicate" );
        require( std::abs( forward - reverse ) < 1.0e-14, "orbit distance depends on argument order" );

        noisy[0] -= 1.0e-4;
        noisy_aligner.align_orbit_closest_to_reference( clean, noisy, noisy_aligned );
        require( distance( clean, noisy_aligned ) > 9.0e-5, "a nearby distinct state was merged with the orbit" );

        // A quarter shift of a substantial odd mode leaves the inversion-odd representation.
        noisy[0] = -2.0;
        noisy[3] = 0.1;
        noisy_aligner.align_orbit_closest_to_reference( clean, noisy, noisy_aligned );
        require( distance( clean, noisy_aligned ) > 1.0, "an incompatible translation was treated as a symmetry" );
        noisy_aligner.align_orbit_closest_to_reference( noisy, clean, noisy_aligned );
        require( distance( noisy, noisy_aligned ) > 1.0, "reverse alignment bypassed the subspace check" );

        noisy[3] = 0.0;
        test_vector compatible_tangent( noisy_operations ), incompatible_tangent( noisy_operations ),
            tangent_result( noisy_operations );
        compatible_tangent[0] = -0.3;
        compatible_tangent[1] = -0.2;
        noisy_aligner.align_orbit_and_tangent_closest_to_reference(
            clean, noisy, compatible_tangent, noisy_aligned, tangent_result
        );
        require( distance( clean, noisy_aligned ) < 1e-14, "compatible tangent prevents state alignment" );
        require( std::abs( tangent_result[0] - 0.3 ) < 1e-14, "tangent used a different residual character" );

        incompatible_tangent[3] = 1.0;
        bool rejected           = false;
        try
        {
            noisy_aligner.apply_last_alignment( incompatible_tangent, tangent_result );
        }
        catch ( const std::domain_error & )
        {
            rejected = true;
        }
        require( rejected, "an unsupported tangent was silently aligned" );
        noisy_aligner.align_orbit_and_tangent_closest_to_reference(
            clean, noisy, incompatible_tangent, noisy_aligned, tangent_result
        );
        require( distance( clean, noisy_aligned ) > 1.0, "joint alignment ignored the tangent subspace" );
        require( distance( incompatible_tangent, tangent_result ) == 0.0, "identity changed the tangent" );

        noisy_aligner.align_orbit_closest_to_reference( clean, noisy, noisy );
        require( distance( clean, noisy ) < 1e-14, "in-place alignment overwrote its input" );
        {
            const auto scope = noisy_aligner.prepare_reference( clean );
            noisy_aligner.align_orbit_closest_to_reference( clean, noisy, noisy_aligned );
            require( distance( clean, noisy_aligned ) < 1e-14, "prepared reference changed alignment" );
        }
        clean[0] = -2.;
        noisy_aligner.align_orbit_closest_to_reference( clean, noisy, noisy_aligned );
        require( distance( clean, noisy_aligned ) < 1e-14, "reference metadata survived its scope" );
        clean[0] = 2.;
        noisy[3] = std::numeric_limits<double>::quiet_NaN();
        rejected = false;
        try
        {
            const auto scope = noisy_aligner.prepare_reference( clean );
            noisy_aligner.align_orbit_closest_to_reference( clean, noisy, noisy_aligned );
        }
        catch ( const std::invalid_argument & )
        {
            rejected = true;
        }
        require( rejected && distance( clean, noisy_aligned ) > 1.0, "invalid alignment modified its destination" );
        noisy[3]         = 0.;
        const auto scope = noisy_aligner.prepare_reference( clean );
        noisy_aligner.align_orbit_closest_to_reference( clean, noisy, noisy_aligned );
        require( distance( clean, noisy_aligned ) < 1e-14, "failed alignment left an active reference scope" );
    }
    catch ( const std::exception &error )
    {
        std::cerr << "FAILED: " << error.what() << '\n';
        return 1;
    }
    std::cout << "PASSED\n";
    return 0;
}
