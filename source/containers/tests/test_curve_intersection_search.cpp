#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <containers/bifurcation_diagram/curve_intersection_search.h>
#include <containers/bifurcation_diagram/curve_point.h>

namespace
{

struct fake_vector_operations
{
    using scalar_type = double;
    using vector_type = double;

    void assign_mul(
        const double left_scale,
        const double& left,
        const double right_scale,
        const double& right,
        double& output)
    {
        output = left_scale*left + right_scale*right;
    }

    void assign(const double& source, double& destination)
    {
        destination = source;
    }
};

struct fake_interpolator
{
    using point_type = container::complex_values<double>;

    bool evaluate_segment_at_lambda(
        const std::vector<point_type>&,
        int lower_index,
        int upper_index,
        double lambda,
        bool,
        const std::string&,
        double& output)
    {
        if(lower_index != 0 || upper_index != 1)
        {
            return false;
        }
        output = 1.0 - lambda;
        return true;
    }
};

void require(bool condition, const std::string& message)
{
    if(!condition)
    {
        throw std::runtime_error(message);
    }
}

bool close_value(double left, double right)
{
    return std::abs(left - right) < 1.0e-13;
}

}

int main()
{
    using point_type = container::complex_values<double>;
    using search_type = container::curve_intersection_search<
        fake_vector_operations,
        fake_interpolator,
        point_type>;

    try
    {
        fake_vector_operations vector_operations;
        fake_interpolator interpolator;
        std::vector<point_type> points(3);
        points[0].lambda = 0.0;
        points[0].vector_norms = {1.0};
        points[0].point_index = 0;
        points[0].segment_id = 1;
        points[0].semicurve_id = 1;
        points[1].lambda = 1.0;
        points[1].vector_norms = {0.0};
        points[1].point_index = 1;
        points[1].segment_id = 1;
        points[1].semicurve_id = 1;
        points[2].lambda = 2.0;
        points[2].vector_norms = {4.0};
        points[2].point_index = 100;
        points[2].segment_id = 2;
        points[2].semicurve_id = 2;

        std::vector<uint64_t> incomplete_segments;
        double first_work = 0.0;
        double second_work = 0.0;
        search_type search;
        search.bind(
            &vector_operations,
            &interpolator,
            &points,
            &incomplete_segments,
            &first_work,
            &second_work);

        container::branch_intersection_policy<double> branch_policy;
        branch_policy.enabled = true;
        branch_policy.signature_tolerance = 1.0e-12;
        branch_policy.state_tolerance = 1.0e-12;
        double hit = -1.0;
        container::branch_intersection_result<double> result;
        const auto distance = [](double left, double right)
        {
            return std::abs(left - right);
        };
        require(
            search.find_branch_intersection(
                0.0,
                0.0,
                1.0,
                1.0,
                {0.0},
                {1.0},
                branch_policy,
                true,
                "unused",
                7,
                hit,
                result,
                distance),
            "known branch intersection");
        require(close_value(result.lambda, 0.5), "known branch lambda");
        require(close_value(hit, 0.5), "known branch state");
        require(result.curve_number == 7, "known branch curve number");

        container::self_intersection_policy<double> self_policy;
        self_policy.enabled = true;
        self_policy.signature_tolerance = 1.0e-12;
        self_policy.state_tolerance = 1.0e-12;
        self_policy.minimum_index_gap = 10;
        result = {};
        require(
            search.find_self_intersection(
                0.0,
                0.0,
                1.0,
                1.0,
                {0.0},
                {1.0},
                self_policy,
                true,
                "unused",
                7,
                hit,
                result,
                distance),
            "self intersection");
        require(result.reason == "self_intersection", "self intersection reason");

        points.resize(2);
        points[0].lambda = 10.0;
        points[0].vector_norms = {-9.0};
        points[0].point_index = 0;
        points[0].segment_id = 4;
        points[0].semicurve_id = 4;
        points[0].endpoint_reason = container::curve_endpoint_reason::none;
        points[1].lambda = 30.0;
        points[1].vector_norms = {-29.0};
        points[1].point_index = 1;
        points[1].segment_id = 4;
        points[1].semicurve_id = 4;
        points[1].endpoint_reason =
            container::curve_endpoint_reason::boundary_max;

        for(const double parameter: {15.0, 19.0, 29.0})
        {
            result = {};
            const double left_parameter = parameter - 0.25;
            const double right_parameter = parameter + 0.25;
            require(
                search.find_branch_intersection(
                    left_parameter,
                    1.0 - left_parameter,
                    right_parameter,
                    1.0 - right_parameter,
                    {1.0 - left_parameter},
                    {1.0 - right_parameter},
                    branch_policy,
                    true,
                    "unused",
                    0,
                    hit,
                    result,
                    distance),
                "boundary-ended analytical segment remains searchable at lambda " +
                    std::to_string(parameter));
        }

        incomplete_segments.push_back(4);
        result = {};
        require(
            search.find_branch_intersection(
                15.0,
                -14.0,
                19.0,
                -18.0,
                {-14.0},
                {-18.0},
                branch_policy,
                true,
                "unused",
                7,
                hit,
                result,
                distance),
            "accepted portion of recoverable segment remains searchable");
    }
    catch(const std::exception& exception)
    {
        std::cerr << "FAILED: " << exception.what() << '\n';
        return 1;
    }

    std::cout << "PASSED\n";
    return 0;
}
