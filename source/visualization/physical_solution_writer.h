#ifndef __VISUALIZATION_PHYSICAL_SOLUTION_WRITER_H__
#define __VISUALIZATION_PHYSICAL_SOLUTION_WRITER_H__

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace visualization
{

namespace detail
{

template<class View>
auto view_data(View& view) -> decltype(view.raw_ptr())
{
    return view.raw_ptr();
}

template<class View>
auto view_data(const View& view) -> decltype(view.raw_ptr())
{
    return view.raw_ptr();
}

template<class View>
auto view_data(View& view) -> decltype(view.data())
{
    return view.data();
}

template<class View>
auto view_data(const View& view) -> decltype(view.data())
{
    return view.data();
}

template<class T>
T* view_data(T* view)
{
    return view;
}

template<class T>
const T* view_data(const T* view)
{
    return view;
}

inline std::string json_escape(const std::string& value)
{
    std::ostringstream stream;
    for(const char ch: value)
    {
        switch(ch)
        {
            case '\\': stream << "\\\\"; break;
            case '"': stream << "\\\""; break;
            case '\n': stream << "\\n"; break;
            case '\r': stream << "\\r"; break;
            case '\t': stream << "\\t"; break;
            default: stream << ch; break;
        }
    }
    return stream.str();
}

template<class VectorOperations>
void write_indexed_vector(
    VectorOperations* vec_ops,
    const typename VectorOperations::vector_type& vec,
    const std::filesystem::path& path,
    const double coordinate_extent,
    const unsigned int precision = 17)
{
    std::ofstream file(path);
    if(!file)
    {
        throw std::runtime_error("visualization: failed to open output file " + path.string());
    }

    const auto host_view = vec_ops->view(vec);
    const auto* data = view_data(host_view);
    const std::size_t size = static_cast<std::size_t>(vec.size());
    file << std::setprecision(precision);
    for(std::size_t i = 0; i < size; ++i)
    {
        const double coordinate =
            coordinate_extent > 0.0 ? coordinate_extent*static_cast<double>(i)/static_cast<double>(size)
                                    : static_cast<double>(i);
        file << coordinate << " " << data[i] << "\n";
    }
}

} // namespace detail

struct visualization_write_result
{
    std::filesystem::path data_file;
    std::string kind;
    std::size_t points = 0;
    std::size_t components = 1;
    double coordinate_extent = 0.0;
};

template<class VectorOperations>
class state_vector_writer
{
public:
    using vector_type = typename VectorOperations::vector_type;

    explicit state_vector_writer(
        VectorOperations* vec_ops_,
        const std::string& kind_ = "state_vector_1d",
        const double coordinate_extent_ = 0.0):
        vec_ops(vec_ops_),
        kind(kind_),
        coordinate_extent(coordinate_extent_)
    {
    }

    visualization_write_result write(
        const vector_type& state,
        const std::filesystem::path& output_prefix,
        const std::size_t,
        const std::size_t)
    {
        visualization_write_result result;
        result.data_file = output_prefix;
        result.data_file += ".dat";
        result.kind = kind;
        result.points = static_cast<std::size_t>(state.size());
        result.coordinate_extent = coordinate_extent;
        detail::write_indexed_vector(vec_ops, state, result.data_file, coordinate_extent);
        return result;
    }

private:
    VectorOperations* vec_ops;
    std::string kind;
    double coordinate_extent;
};

template<class StateVectorOperations, class PhysicalVectorOperations, class NonlinearOperator>
class physical_solution_writer
{
public:
    using state_vector_type = typename StateVectorOperations::vector_type;
    using physical_vector_type = typename PhysicalVectorOperations::vector_type;

    physical_solution_writer(
        PhysicalVectorOperations* physical_vec_ops_,
        NonlinearOperator* nonlinear_operator_,
        const std::size_t physical_size_,
        const double coordinate_extent_):
        physical_vec_ops(physical_vec_ops_),
        nonlinear_operator(nonlinear_operator_),
        physical_size(physical_size_),
        coordinate_extent(coordinate_extent_)
    {
        physical_vec_ops->init_vector(physical);
        physical_vec_ops->start_use_vector(physical);
    }

    physical_solution_writer(const physical_solution_writer&) = delete;
    physical_solution_writer& operator=(const physical_solution_writer&) = delete;

    ~physical_solution_writer()
    {
        physical_vec_ops->stop_use_vector(physical);
        physical_vec_ops->free_vector(physical);
    }

    visualization_write_result write(
        const state_vector_type& state,
        const std::filesystem::path& output_prefix,
        const std::size_t,
        const std::size_t)
    {
        nonlinear_operator->physical_solution(state, physical);

        visualization_write_result result;
        result.data_file = output_prefix;
        result.data_file += ".dat";
        result.kind = "physical_scalar_1d";
        result.points = physical_size;
        result.coordinate_extent = coordinate_extent;
        detail::write_indexed_vector(physical_vec_ops, physical, result.data_file, coordinate_extent);
        return result;
    }

private:
    PhysicalVectorOperations* physical_vec_ops;
    NonlinearOperator* nonlinear_operator;
    std::size_t physical_size;
    double coordinate_extent;
    physical_vector_type physical;
};

} // namespace visualization

#endif
