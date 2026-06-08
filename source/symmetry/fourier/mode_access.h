#ifndef __SYMMETRY_FOURIER_MODE_ACCESS_H__
#define __SYMMETRY_FOURIER_MODE_ACCESS_H__

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace symmetry
{
namespace fourier
{

template <class Complex>
class packed_positive_mode_view
{
public:
    using complex_type = Complex;

    packed_positive_mode_view() = default;

    packed_positive_mode_view( complex_type *data, const std::size_t size ) : data_( data ), size_( size )
    {
        if ( data == nullptr && size != 0 )
            throw std::invalid_argument( "packed_positive_mode_view got null data" );
    }

    explicit packed_positive_mode_view( std::vector<complex_type> &data ) : packed_positive_mode_view( data.data(), data.size() )
    {
    }

    std::size_t size() const
    {
        return size_;
    }

    bool has_mode( const std::size_t mode ) const
    {
        return mode < size_;
    }

    complex_type &mode( const std::size_t mode )
    {
        if ( mode >= size_ )
            throw std::out_of_range( "packed_positive_mode_view::mode" );
        return data_[mode];
    }

    const complex_type &mode( const std::size_t mode ) const
    {
        if ( mode >= size_ )
            throw std::out_of_range( "packed_positive_mode_view::mode const" );
        return data_[mode];
    }

    complex_type *data()
    {
        return data_;
    }

    const complex_type *data() const
    {
        return data_;
    }

private:
    complex_type *data_ = nullptr;
    std::size_t size_ = 0;
};

template <class Complex>
class const_packed_positive_mode_view
{
public:
    using complex_type = Complex;

    const_packed_positive_mode_view() = default;

    const_packed_positive_mode_view( const complex_type *data, const std::size_t size ) : data_( data ), size_( size )
    {
        if ( data == nullptr && size != 0 )
            throw std::invalid_argument( "const_packed_positive_mode_view got null data" );
    }

    explicit const_packed_positive_mode_view( const std::vector<complex_type> &data )
        : const_packed_positive_mode_view( data.data(), data.size() )
    {
    }

    std::size_t size() const
    {
        return size_;
    }

    bool has_mode( const std::size_t mode ) const
    {
        return mode < size_;
    }

    const complex_type &mode( const std::size_t mode ) const
    {
        if ( mode >= size_ )
            throw std::out_of_range( "const_packed_positive_mode_view::mode" );
        return data_[mode];
    }

    const complex_type *data() const
    {
        return data_;
    }

private:
    const complex_type *data_ = nullptr;
    std::size_t size_ = 0;
};

} // namespace fourier
} // namespace symmetry

#endif
