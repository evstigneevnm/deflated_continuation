#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <containers/bifurcation_diagram/curve_interpolator.h>
#include <containers/bifurcation_diagram/curve_point.h>

namespace
{

struct fake_vector_operations
{
    using scalar_type = double;
    using vector_type = double;

    void add_mul(double left_scale, const double& left, double right_scale, double& right)
    {
        right = left_scale*left + right_scale*right;
    }

    void assign(const double& source, double& destination)
    {
        destination = source;
    }
};

struct fake_store
{
    void read(const std::string& directory, uint64_t id, double& value) const
    {
        value = data.at(directory + "/" + std::to_string(id));
    }

    std::unordered_map<std::string, double> data;
};

struct fake_nonlinear_operator
{
};

struct fake_newton
{
    bool solve(fake_nonlinear_operator*, double& value, const double& lambda)
    {
        calls++;
        if(!converges)
        {
            return false;
        }
        value += correction_scale*lambda;
        return true;
    }

    int calls = 0;
    bool converges = true;
    double correction_scale = 0.1;
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
    return std::abs(left - right) < 1.0e-14;
}

}

int main()
{
    using point_type = container::complex_values<double>;
    using interpolator_type = container::curve_interpolator<
        fake_vector_operations,
        fake_store,
        fake_nonlinear_operator,
        fake_newton,
        point_type>;

    try
    {
        fake_vector_operations vector_operations;
        fake_store store;
        fake_nonlinear_operator nonlinear_operator;
        fake_newton newton;
        double lower_work = 0;
        double upper_work = 0;
        interpolator_type interpolator;
        interpolator.bind(
            &vector_operations,
            &store,
            &nonlinear_operator,
            &newton,
            &lower_work,
            &upper_work);

        std::vector<point_type> points(4);
        for(std::size_t index = 0; index < points.size(); ++index)
        {
            points[index].lambda = static_cast<double>(index);
            points[index].segment_id = 2;
        }
        points[0].is_data_avaliable = true;
        points[0].id_file_name = 1;
        points[3].is_data_avaliable = true;
        points[3].id_file_name = 2;
        store.data["curve/1"] = 0.0;
        store.data["curve/2"] = 3.0;

        double output = -1;
        require(
            interpolator.evaluate_segment_at_lambda(points, 1, 2, 1.5, true, "curve", output),
            "interpolated solve");
        require(close_value(output, 1.65), "interpolated value");
        require(newton.calls == 1, "interpolated Newton call");

        output = -1;
        require(
            interpolator.evaluate_segment_at_lambda(points, 0, 1, 0.0, true, "curve", output),
            "saved endpoint");
        require(close_value(output, 0.0), "saved endpoint value");
        require(newton.calls == 1, "saved endpoint bypasses Newton");

        points[3].segment_id = 3;
        require(
            !interpolator.evaluate_segment_at_lambda(points, 1, 2, 1.5, true, "curve", output),
            "segment boundary rejection");
        points[3].segment_id = 2;

        newton.converges = false;
        require(
            !interpolator.evaluate_segment_at_lambda(points, 1, 2, 1.5, true, "curve", output),
            "Newton failure propagation");
    }
    catch(const std::exception& exception)
    {
        std::cerr << "FAILED: " << exception.what() << '\n';
        return 1;
    }
    std::cout << "PASSED\n";
    return 0;
}
