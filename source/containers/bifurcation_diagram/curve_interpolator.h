#ifndef __BIFURCATION_DIAGRAM_CURVE_INTERPOLATOR_H__
#define __BIFURCATION_DIAGRAM_CURVE_INTERPOLATOR_H__

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace container
{

template<
    class VectorOperations,
    class VectorStore,
    class NonlinearOperator,
    class Newton,
    class Point>
class curve_interpolator
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;

    void bind(
        VectorOperations* vector_operations,
        VectorStore* vector_store,
        NonlinearOperator* nonlinear_operator,
        Newton* newton,
        vector_type* lower_work,
        vector_type* upper_work)
    {
        vector_operations_ = vector_operations;
        vector_store_ = vector_store;
        nonlinear_operator_ = nonlinear_operator;
        newton_ = newton;
        lower_work_ = lower_work;
        upper_work_ = upper_work;
    }

    bool read_saved_point(
        const std::string& directory,
        const Point& point,
        vector_type& output) const
    {
        if(!point.is_data_avaliable)
        {
            return false;
        }
        vector_store_->read(directory, point.id_file_name, output);
        return true;
    }

    bool load_lower(
        const std::vector<Point>& points,
        int index,
        const uint64_t segment_id,
        const bool segment_metadata_available,
        const std::string& directory)
    {
        while(index >= 0)
        {
            const auto& point = points[static_cast<std::size_t>(index)];
            if(segment_metadata_available && point.segment_id != segment_id)
            {
                break;
            }
            if(point.is_data_avaliable)
            {
                lower_lambda_ = point.lambda;
                vector_store_->read(directory, point.id_file_name, *lower_work_);
                return true;
            }
            --index;
        }
        return false;
    }

    bool load_upper(
        const std::vector<Point>& points,
        int index,
        const uint64_t segment_id,
        const bool segment_metadata_available,
        const std::string& directory)
    {
        const int point_count = static_cast<int>(points.size());
        while(index < point_count)
        {
            const auto& point = points[static_cast<std::size_t>(index)];
            if(segment_metadata_available && point.segment_id != segment_id)
            {
                break;
            }
            if(point.is_data_avaliable)
            {
                upper_lambda_ = point.lambda;
                vector_store_->read(directory, point.id_file_name, *upper_work_);
                return true;
            }
            ++index;
        }
        return false;
    }

    bool interpolate_prepared(const scalar_type& lambda)
    {
        const scalar_type weight = (lambda - lower_lambda_)/(upper_lambda_ - lower_lambda_);
        vector_operations_->add_mul(
            scalar_type(1) - weight,
            *lower_work_,
            weight,
            *upper_work_);
        upper_lambda_ = lambda;
        return newton_->solve(nonlinear_operator_, *upper_work_, lambda);
    }

    bool interpolate_segment(
        const std::vector<Point>& points,
        const int lower_index,
        const int upper_index,
        const scalar_type& lambda,
        const bool segment_metadata_available,
        const std::string& directory,
        vector_type& output)
    {
        const auto& lower = points[static_cast<std::size_t>(lower_index)];
        const auto& upper = points[static_cast<std::size_t>(upper_index)];
        if(!load_lower(points, lower_index, lower.segment_id, segment_metadata_available, directory) ||
           !load_upper(points, upper_index, upper.segment_id, segment_metadata_available, directory) ||
           !interpolate_prepared(lambda))
        {
            return false;
        }
        vector_operations_->assign(*upper_work_, output);
        return true;
    }

    bool evaluate_segment_at_lambda(
        const std::vector<Point>& points,
        const int lower_index,
        const int upper_index,
        const scalar_type& lambda,
        const bool segment_metadata_available,
        const std::string& directory,
        vector_type& output)
    {
        const auto& lower = points[static_cast<std::size_t>(lower_index)];
        const auto& upper = points[static_cast<std::size_t>(upper_index)];
        if(same_scalar(lower.lambda, lambda) && read_saved_point(directory, lower, output))
        {
            return true;
        }
        if(same_scalar(upper.lambda, lambda) && read_saved_point(directory, upper, output))
        {
            return true;
        }
        return interpolate_segment(
            points,
            lower_index,
            upper_index,
            lambda,
            segment_metadata_available,
            directory,
            output);
    }

private:
    static scalar_type scalar_abs(const scalar_type& value)
    {
        return value < scalar_type(0) ? -value : value;
    }

    static bool same_scalar(const scalar_type& left, const scalar_type& right)
    {
        const scalar_type scale = std::max<scalar_type>(
            scalar_type(1),
            std::max<scalar_type>(scalar_abs(left), scalar_abs(right)));
        return scalar_abs(left - right) <=
            scalar_type(64)*std::numeric_limits<scalar_type>::epsilon()*scale;
    }

    VectorOperations* vector_operations_ = nullptr;
    VectorStore* vector_store_ = nullptr;
    NonlinearOperator* nonlinear_operator_ = nullptr;
    Newton* newton_ = nullptr;
    vector_type* lower_work_ = nullptr;
    vector_type* upper_work_ = nullptr;
    scalar_type lower_lambda_ = scalar_type(0);
    scalar_type upper_lambda_ = scalar_type(0);
};

} // namespace container

#endif // __BIFURCATION_DIAGRAM_CURVE_INTERPOLATOR_H__
