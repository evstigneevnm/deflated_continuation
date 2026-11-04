#ifndef PERIODIC_ORBIT_AFFINE_SECTION_H
#define PERIODIC_ORBIT_AFFINE_SECTION_H

#include <memory>
#include <stdexcept>
#include <utility>
#include <nmfd/detail/vector_wrap.h>
#include <scfd/utils/scalar_traits.h>

namespace periodic_orbit
{
template <class Space>
struct euclidean_section_metric
{
    using T = typename Space::norm_type;
    using X = typename Space::vector_type;
    std::shared_ptr<Space> space;
    explicit euclidean_section_metric( std::shared_ptr<Space> operations ) : space( std::move( operations ) )
    {
        if ( !space )
            throw std::invalid_argument( "Missing section metric space" );
    }
    T product( const X &a, const X &b ) const
    {
        return scfd::utils::scalar_traits<typename Space::scalar_type>::real( space->scalar_prod( a, b ) );
    }
    T norm( const X &x ) const
    {
        return scfd::utils::scalar_traits<T>::sqrt( product( x, x ) );
    }
};

template <class Space, class Metric = euclidean_section_metric<Space>>
class affine_section
{
public:
    using scalar_type  = typename Space::norm_type;
    using vector_type  = typename Space::vector_type;
    using metric_type  = Metric;
    using T            = scalar_type;
    using X            = vector_type;
    using storage_type = nmfd::detail::vector_wrap<Space, false, false>;

    struct evaluation_workspace
    {
        std::shared_ptr<Space> space;
        storage_type           difference;
        explicit evaluation_workspace( std::shared_ptr<Space> operations )
            : space( operations ? std::move( operations ) : throw std::invalid_argument( "Missing section space" ) ),
              difference( *space )
        {
            difference.init();
            difference.start_use();
        }
    };

private:
    struct geometry
    {
        std::shared_ptr<Space>  space;
        std::shared_ptr<Metric> metric;
        storage_type            center, normal;
        geometry( std::shared_ptr<Space> operations, std::shared_ptr<Metric> measure, const X &c, const X &n )
            : space( operations ? std::move( operations ) : throw std::invalid_argument( "Missing section space" ) ),
              metric( measure ? std::move( measure ) : throw std::invalid_argument( "Missing section metric" ) ),
              center( *space ), normal( *space )
        {
            if ( space->get_size( c ) != space->get_vector_size() || space->get_size( n ) != space->get_vector_size() ||
                 !space->is_valid_number( c ) || !space->is_valid_number( n ) )
                throw std::invalid_argument( "Invalid section center/normal" );
            const T length = metric->norm( n );
            if ( !scfd::utils::scalar_traits<T>::is_finite( length ) || length <= T( 0 ) )
                throw std::invalid_argument( "Section normal must be nonzero and finite" );
            center.init();
            center.start_use();
            normal.init();
            normal.start_use();
            space->assign( c, *center );
            space->assign_lin_comb( typename Space::scalar_type( T( 1 ) / length ), n, *normal );
        }
    };
    std::shared_ptr<const geometry> data_;

public:
    class evaluator
    {
        std::shared_ptr<const geometry> data_;
        evaluation_workspace           *work_;
        friend class affine_section;
        evaluator( std::shared_ptr<const geometry> data, evaluation_workspace &work )
            : data_( std::move( data ) ), work_( &work )
        {
        }

    public:
        T value( T, const X &state ) const
        {
            // Form the difference before reducing; subtracting two large dots loses accuracy.
            data_->space->assign_lin_comb( 1, state, -1, *data_->center, *work_->difference );
            return data_->metric->product( *data_->normal, *work_->difference );
        }
        T state_derivative( T, const X &, const X &v ) const
        {
            return data_->metric->product( *data_->normal, v );
        }
        T distance( const X &state ) const
        {
            data_->space->assign_lin_comb( 1, state, -1, *data_->center, *work_->difference );
            return data_->metric->norm( *work_->difference );
        }
    };

    affine_section( std::shared_ptr<Space> space, const X &center, const X &normal )
        : affine_section( space, center, normal, std::make_shared<Metric>( space ) )
    {
    }
    affine_section( std::shared_ptr<Space> space, const X &center, const X &normal, std::shared_ptr<Metric> metric )
        : data_( std::make_shared<geometry>( std::move( space ), std::move( metric ), center, normal ) )
    {
    }
    evaluator bind( evaluation_workspace &work ) const
    {
        if ( work.space.get() != data_->space.get() )
            throw std::invalid_argument( "Section evaluation workspace belongs to another space" );
        return evaluator( data_, work );
    }
    const X &center() const
    {
        return *data_->center;
    }
    const X &normal() const
    {
        return *data_->normal;
    }
};
}
#endif
