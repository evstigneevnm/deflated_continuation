#ifndef __NMFD_OPERATIONS_IO_MATRIX_MARKET_H__
#define __NMFD_OPERATIONS_IO_MATRIX_MARKET_H__

#include <algorithm>
#include <cctype>
#include <complex>
#include <cstddef>
#include <fstream>
#include <istream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <nmfd/operations/sparse/host_csr_matrix.h>

namespace nmfd
{
namespace operations
{
namespace io
{

enum class matrix_market_field
{
    real,
    integer,
    complex,
    pattern
};

enum class matrix_market_symmetry
{
    general,
    symmetric,
    skew_symmetric,
    hermitian
};

struct matrix_market_metadata
{
    std::size_t rows = 0;
    std::size_t columns = 0;
    std::size_t stored_entries = 0;
    matrix_market_field field = matrix_market_field::real;
    matrix_market_symmetry symmetry = matrix_market_symmetry::general;
};

template<class Scalar, class Index = std::size_t>
struct matrix_market_coordinate_matrix
{
    using scalar_type = Scalar;
    using index_type = Index;
    using entry_type =
        nmfd::operations::sparse::coordinate_entry<scalar_type, index_type>;

    matrix_market_metadata metadata;
    std::vector<entry_type> entries;

    nmfd::operations::sparse::host_csr_matrix<scalar_type, index_type>
    to_host_csr(bool drop_merged_zeros = true) const
    {
        return {
            checked_index(metadata.rows),
            checked_index(metadata.columns),
            entries,
            drop_merged_zeros};
    }

private:
    static index_type checked_index(std::size_t value)
    {
        if(value > static_cast<std::size_t>(
                       std::numeric_limits<index_type>::max()))
        {
            throw std::overflow_error(
                "Matrix Market dimension does not fit the requested index type");
        }
        return static_cast<index_type>(value);
    }
};

namespace detail
{

template<class T>
struct is_std_complex : std::false_type
{
};

template<class T>
struct is_std_complex<std::complex<T>> : std::true_type
{
};

inline std::string lower_copy(std::string value)
{
    std::transform(
        value.begin(),
        value.end(),
        value.begin(),
        [](unsigned char character)
        {
            return static_cast<char>(std::tolower(character));
        });
    return value;
}

inline bool blank(const std::string& line)
{
    return std::all_of(
        line.begin(),
        line.end(),
        [](unsigned char character)
        {
            return std::isspace(character) != 0;
        });
}

inline bool comment(const std::string& line)
{
    const auto first = std::find_if_not(
        line.begin(),
        line.end(),
        [](unsigned char character)
        {
            return std::isspace(character) != 0;
        });
    return first != line.end() && *first == '%';
}

inline bool next_data_line(
    std::istream& input,
    std::string& line,
    std::size_t& line_number)
{
    while(std::getline(input, line))
    {
        ++line_number;
        if(!blank(line) && !comment(line))
            return true;
    }
    return false;
}

inline std::runtime_error parse_error(
    std::size_t line,
    const std::string& message)
{
    return std::runtime_error(
        "Matrix Market parse error at line " +
        std::to_string(line) + ": " + message);
}

inline void require_no_extra_tokens(
    std::istringstream& parser,
    std::size_t line_number)
{
    std::string extra;
    if(parser >> extra)
        throw parse_error(line_number, "unexpected trailing token '" + extra + "'");
}

inline matrix_market_field parse_field(
    const std::string& token,
    std::size_t line_number)
{
    const auto normalized = lower_copy(token);
    if(normalized == "real")
        return matrix_market_field::real;
    if(normalized == "integer")
        return matrix_market_field::integer;
    if(normalized == "complex")
        return matrix_market_field::complex;
    if(normalized == "pattern")
        return matrix_market_field::pattern;
    throw parse_error(line_number, "unsupported field '" + token + "'");
}

inline matrix_market_symmetry parse_symmetry(
    const std::string& token,
    std::size_t line_number)
{
    const auto normalized = lower_copy(token);
    if(normalized == "general")
        return matrix_market_symmetry::general;
    if(normalized == "symmetric")
        return matrix_market_symmetry::symmetric;
    if(normalized == "skew-symmetric")
        return matrix_market_symmetry::skew_symmetric;
    if(normalized == "hermitian")
        return matrix_market_symmetry::hermitian;
    throw parse_error(line_number, "unsupported symmetry '" + token + "'");
}

template<class Scalar>
typename std::enable_if<!is_std_complex<Scalar>::value, Scalar>::type
make_scalar(
    long double real,
    long double imaginary,
    std::size_t line_number)
{
    if(imaginary != 0.0L)
        throw parse_error(
            line_number,
            "complex entry cannot be represented by a real scalar type");
    return static_cast<Scalar>(real);
}

template<class Scalar>
typename std::enable_if<is_std_complex<Scalar>::value, Scalar>::type
make_scalar(
    long double real,
    long double imaginary,
    std::size_t)
{
    using value_type = typename Scalar::value_type;
    return Scalar(
        static_cast<value_type>(real),
        static_cast<value_type>(imaginary));
}

template<class Scalar>
typename std::enable_if<!is_std_complex<Scalar>::value, Scalar>::type
conjugate(const Scalar& value)
{
    return value;
}

template<class Scalar>
typename std::enable_if<is_std_complex<Scalar>::value, Scalar>::type
conjugate(const Scalar& value)
{
    return std::conj(value);
}

template<class Index>
Index checked_zero_based_index(
    long long one_based,
    std::size_t bound,
    const char* name,
    std::size_t line_number)
{
    if(one_based < 1 || static_cast<unsigned long long>(one_based) > bound)
        throw parse_error(
            line_number,
            std::string(name) + " index is outside matrix bounds");
    const auto zero_based =
        static_cast<unsigned long long>(one_based - 1);
    if(zero_based > static_cast<unsigned long long>(
                        std::numeric_limits<Index>::max()))
    {
        throw parse_error(
            line_number,
            std::string(name) + " index does not fit the requested index type");
    }
    return static_cast<Index>(zero_based);
}

} // namespace detail

template<class Scalar, class Index = std::size_t>
matrix_market_coordinate_matrix<Scalar, Index>
read_matrix_market(std::istream& input)
{
    using matrix_type = matrix_market_coordinate_matrix<Scalar, Index>;
    using entry_type = typename matrix_type::entry_type;

    matrix_type result;
    std::string line;
    std::size_t line_number = 0;
    if(!std::getline(input, line))
        throw std::runtime_error("Matrix Market input is empty");
    ++line_number;

    std::istringstream banner(line);
    std::string identifier;
    std::string object;
    std::string format;
    std::string field;
    std::string symmetry;
    if(!(banner >> identifier >> object >> format >> field >> symmetry))
        throw detail::parse_error(line_number, "incomplete banner");
    detail::require_no_extra_tokens(banner, line_number);
    if(detail::lower_copy(identifier) != "%%matrixmarket")
        throw detail::parse_error(line_number, "missing %%MatrixMarket banner");
    if(detail::lower_copy(object) != "matrix")
        throw detail::parse_error(line_number, "only matrix objects are supported");
    if(detail::lower_copy(format) != "coordinate")
        throw detail::parse_error(
            line_number,
            "only coordinate matrices are supported");

    result.metadata.field = detail::parse_field(field, line_number);
    result.metadata.symmetry =
        detail::parse_symmetry(symmetry, line_number);
    if(
        result.metadata.field == matrix_market_field::complex &&
        !detail::is_std_complex<Scalar>::value)
    {
        throw detail::parse_error(
            line_number,
            "complex matrices require std::complex scalar storage");
    }

    if(!detail::next_data_line(input, line, line_number))
        throw detail::parse_error(line_number, "missing size record");
    std::istringstream dimensions(line);
    unsigned long long rows = 0;
    unsigned long long columns = 0;
    unsigned long long stored_entries = 0;
    if(!(dimensions >> rows >> columns >> stored_entries))
        throw detail::parse_error(line_number, "invalid size record");
    detail::require_no_extra_tokens(dimensions, line_number);
    if(
        rows > std::numeric_limits<std::size_t>::max() ||
        columns > std::numeric_limits<std::size_t>::max() ||
        stored_entries > std::numeric_limits<std::size_t>::max())
    {
        throw detail::parse_error(
            line_number,
            "matrix dimensions exceed host size type");
    }

    result.metadata.rows = static_cast<std::size_t>(rows);
    result.metadata.columns = static_cast<std::size_t>(columns);
    result.metadata.stored_entries =
        static_cast<std::size_t>(stored_entries);
    if(
        result.metadata.symmetry != matrix_market_symmetry::general &&
        result.metadata.rows != result.metadata.columns)
    {
        throw detail::parse_error(
            line_number,
            "a structured matrix must be square");
    }

    const bool structured =
        result.metadata.symmetry != matrix_market_symmetry::general;
    result.entries.reserve(
        result.metadata.stored_entries * (structured ? 2 : 1));

    for(std::size_t at = 0; at < result.metadata.stored_entries; ++at)
    {
        if(!detail::next_data_line(input, line, line_number))
            throw detail::parse_error(
                line_number,
                "fewer entries than declared in the size record");

        std::istringstream entry_parser(line);
        long long one_based_row = 0;
        long long one_based_column = 0;
        if(!(entry_parser >> one_based_row >> one_based_column))
            throw detail::parse_error(line_number, "invalid coordinate record");

        long double real = 1.0L;
        long double imaginary = 0.0L;
        switch(result.metadata.field)
        {
        case matrix_market_field::real:
            if(!(entry_parser >> real))
                throw detail::parse_error(
                    line_number,
                    "real coordinate is missing its value");
            break;
        case matrix_market_field::integer:
        {
            long long integer = 0;
            if(!(entry_parser >> integer))
                throw detail::parse_error(
                    line_number,
                    "integer coordinate is missing its value");
            real = static_cast<long double>(integer);
            break;
        }
        case matrix_market_field::complex:
            if(!(entry_parser >> real >> imaginary))
                throw detail::parse_error(
                    line_number,
                    "complex coordinate requires real and imaginary values");
            break;
        case matrix_market_field::pattern:
            break;
        }
        detail::require_no_extra_tokens(entry_parser, line_number);

        const auto row = detail::checked_zero_based_index<Index>(
            one_based_row,
            result.metadata.rows,
            "row",
            line_number);
        const auto column = detail::checked_zero_based_index<Index>(
            one_based_column,
            result.metadata.columns,
            "column",
            line_number);
        const Scalar value =
            detail::make_scalar<Scalar>(real, imaginary, line_number);

        if(
            result.metadata.symmetry ==
                matrix_market_symmetry::skew_symmetric &&
            row == column &&
            value != Scalar{})
        {
            throw detail::parse_error(
                line_number,
                "skew-symmetric diagonal entry must be zero");
        }
        if(
            result.metadata.symmetry ==
                matrix_market_symmetry::hermitian &&
            row == column &&
            value != detail::conjugate(value))
        {
            throw detail::parse_error(
                line_number,
                "Hermitian diagonal entry must be real");
        }

        result.entries.push_back(entry_type{row, column, value});
        if(row == column || !structured)
            continue;

        Scalar reflected = value;
        if(
            result.metadata.symmetry ==
            matrix_market_symmetry::skew_symmetric)
        {
            reflected = -value;
        }
        else if(
            result.metadata.symmetry ==
            matrix_market_symmetry::hermitian)
        {
            reflected = detail::conjugate(value);
        }
        result.entries.push_back(entry_type{column, row, reflected});
    }

    if(detail::next_data_line(input, line, line_number))
        throw detail::parse_error(
            line_number,
            "more entries than declared in the size record");
    return result;
}

template<class Scalar, class Index = std::size_t>
matrix_market_coordinate_matrix<Scalar, Index>
read_matrix_market_file(const std::string& path)
{
    std::ifstream input(path);
    if(!input)
        throw std::runtime_error(
            "Unable to open Matrix Market file: " + path);
    return read_matrix_market<Scalar, Index>(input);
}

} // namespace io
} // namespace operations
} // namespace nmfd

#endif
