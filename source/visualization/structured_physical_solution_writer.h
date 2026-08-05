#ifndef __VISUALIZATION_STRUCTURED_PHYSICAL_SOLUTION_WRITER_H__
#define __VISUALIZATION_STRUCTURED_PHYSICAL_SOLUTION_WRITER_H__

#include <array>
#include <cstddef>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include <visualization/io/numpy_array_writer.h>
#include <visualization/physical_solution_writer.h>

namespace visualization
{

template<
    class StateVectorOperations,
    class PhysicalVectorOperations,
    class NonlinearOperator,
    std::size_t Dimension>
class structured_physical_solution_writer
{
public:
    using state_vector_type = typename StateVectorOperations::vector_type;
    using physical_vector_type = typename PhysicalVectorOperations::vector_type;
    using scalar_type = typename PhysicalVectorOperations::scalar_type;
    using shape_type = std::array<std::size_t, Dimension>;
    using coordinates_type = std::array<double, Dimension>;
    using periodicity_type = std::array<bool, Dimension>;
    using axis_names_type = std::array<std::string, Dimension>;

    structured_physical_solution_writer(
        PhysicalVectorOperations* physical_vec_ops_,
        NonlinearOperator* nonlinear_operator_,
        const shape_type& shape_,
        const coordinates_type& origin_,
        const coordinates_type& lengths_,
        const periodicity_type& periodic_,
        const axis_names_type& axis_names_,
        std::string field_name_,
        std::string producer_backend_):
        physical_vec_ops(require_pointer(physical_vec_ops_, "physical vector operations")),
        nonlinear_operator(require_pointer(nonlinear_operator_, "nonlinear operator")),
        shape(shape_),
        origin(origin_),
        lengths(lengths_),
        periodic(periodic_),
        axis_names(axis_names_),
        field_name(std::move(field_name_)),
        producer_backend(std::move(producer_backend_)),
        physical_size(element_count(shape_))
    {
        physical_vec_ops->init_vector(physical);
        physical_vec_ops->start_use_vector(physical);
        if(physical_vec_ops->get_size(physical) != physical_size)
        {
            physical_vec_ops->stop_use_vector(physical);
            physical_vec_ops->free_vector(physical);
            throw std::invalid_argument("structured visualization vector size does not match its shape");
        }
    }

    structured_physical_solution_writer(const structured_physical_solution_writer&) = delete;
    structured_physical_solution_writer& operator=(const structured_physical_solution_writer&) = delete;

    ~structured_physical_solution_writer()
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
        const auto host_view = physical_vec_ops->view(physical);
        const auto* data = detail::view_data(host_view);

        visualization_write_result result;
        result.data_file = output_prefix;
        result.data_file += ".npy";
        io::write_numpy_array(result.data_file, data, shape);

        result.kind = "physical_scalar_" + std::to_string(Dimension) + "d";
        result.format = "npy";
        result.field_name = field_name;
        result.scalar_type = io::numpy_scalar_name<scalar_type>();
        result.storage_order = "last_index_fast";
        result.producer_backend = producer_backend;
        result.points = physical_size;
        result.components = 1;
        result.shape.assign(shape.begin(), shape.end());
        result.origin.assign(origin.begin(), origin.end());
        result.periodic.assign(periodic.begin(), periodic.end());
        result.axis_names.assign(axis_names.begin(), axis_names.end());
        result.spacing.reserve(Dimension);
        for(std::size_t dimension = 0; dimension < Dimension; ++dimension)
        {
            result.spacing.push_back(lengths[dimension]/static_cast<double>(shape[dimension]));
        }
        return result;
    }

private:
    template<class Pointer>
    static Pointer* require_pointer(Pointer* pointer, const char* label)
    {
        if(pointer == nullptr)
        {
            throw std::invalid_argument(std::string("structured visualization requires ") + label);
        }
        return pointer;
    }

    static std::size_t element_count(const shape_type& value)
    {
        std::size_t result = 1;
        for(const std::size_t extent: value)
        {
            if(extent == 0)
            {
                throw std::invalid_argument("structured visualization dimensions must be positive");
            }
            if(extent > std::numeric_limits<std::size_t>::max()/result)
            {
                throw std::overflow_error("structured visualization size overflows size_t");
            }
            result *= extent;
        }
        return result;
    }

    PhysicalVectorOperations* physical_vec_ops;
    NonlinearOperator* nonlinear_operator;
    shape_type shape;
    coordinates_type origin;
    coordinates_type lengths;
    periodicity_type periodic;
    axis_names_type axis_names;
    std::string field_name;
    std::string producer_backend;
    std::size_t physical_size;
    physical_vector_type physical;
};

} // namespace visualization

#endif
