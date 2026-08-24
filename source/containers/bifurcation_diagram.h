#ifndef __BIFURCATION_DIAGRAM_H__
#define __BIFURCATION_DIAGRAM_H__

/**
*
*   Main container that holds all bifurcaiton diagram curves.
*   
*   The curves are typed by the Curve template parameter.
*   Can be used with arbitrary curves, but inlined curves are more preferable.
*   Helper class is also included to avoid memory usage on the curves during interpolation
*/

#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <type_traits>
#include <utility>
//using boost for serialization
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/split_member.hpp>

#include <containers/branch_intersection.h>
#include <containers/intersection_status.h>
#include <containers/bifurcation_diagram/symmetry_archive_audit.h>

namespace container
{

namespace bifurcation_diagram_detail
{

template<class NonlinearOperator, class = void>
struct has_norm_labels: std::false_type
{
};

template<class NonlinearOperator>
struct has_norm_labels<
    NonlinearOperator,
    std::void_t<decltype(std::declval<const NonlinearOperator&>().norm_bifurcation_diagram_labels())>
>: std::true_type
{
};

template<class NonlinearOperator>
std::vector<std::string> norm_labels(const NonlinearOperator* nonlin_op)
{
    if constexpr(has_norm_labels<NonlinearOperator>::value)
    {
        return nonlin_op->norm_bifurcation_diagram_labels();
    }
    else
    {
        return {};
    }
}

}

template<class VectorOperations, class VectorFileOperations, class Log, class NonlinearOperator, class Newton, class SolutionStorage,  class Curve, class CurveHelper>
class bifurcation_diagram
{
private:
    friend class boost::serialization::access;

    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;

    typedef CurveHelper cont_help_t;

    VectorOperations* vec_ops = nullptr;
    VectorFileOperations* file_ops = nullptr;
    Log* log = nullptr;
    NonlinearOperator* nonlin_op = nullptr;
    Newton* newton = nullptr;
    cont_help_t* cont_help = nullptr;
    unsigned int skip_output = 10;
    std::string directory;
    bool legend_written = false;

public:
    typedef typename Curve::values_t curve_point_type;
    typedef typename Curve::symmetry_event_record_type symmetry_event_record_type;

public:
    bifurcation_diagram(VectorOperations* vec_ops_, VectorFileOperations* vec_files_, Log* log_, NonlinearOperator* nlin_op_, Newton* newton_, const std::string& directory_ = {}, unsigned int skip_output_ = 10):
    vec_ops(vec_ops_),
    file_ops(vec_files_),
    log(log_),
    nonlin_op(nlin_op_),
    newton(newton_),
    skip_output(skip_output_),
    directory(directory_)
    {
        cont_help = new cont_help_t(vec_ops);
        curve_number = -1;
        write_legend_file();
    }

    bifurcation_diagram()
    {
        //void default constructor for boost serialization
        log = nullptr;
    }

    Log* get_log()const
    {
        return log;
    }

    ~bifurcation_diagram()
    {
        delete cont_help;
    }

    void set_skip_output(unsigned int skip_output_)
    {
        skip_output = skip_output_;
    }

    void get_current_ref(Curve*& curve_ref)
    {
        //std::cout << "reference to the curve inside = " << &curve_container.back() << std::endl;
        curve_container.back().set_main_refs( vec_ops, file_ops, log, nonlin_op, newton, cont_help );
        curve_ref = &curve_container.back();
    }

    Curve* get_current_ref()
    {
        //std::cout << "reference to the curve inside = " << &curve_container.back() << std::endl;
        curve_container.back().set_main_refs( vec_ops, file_ops, log, nonlin_op, newton, cont_help );
        return (&curve_container.back());
    }

    void reset_curve_output_directories()
    {
        for(auto& curve: curve_container)
        {
            curve.rebind_runtime_directory(
                vec_ops,
                file_ops,
                log,
                nonlin_op,
                newton,
                cont_help,
                directory);
        }
    }

    std::size_t current_curve() const
    {
        return curve_container.size();
    }

    std::size_t curve_count() const
    {
        return curve_container.size();
    }

    bool restore_analytical_curve_provenance(
        const std::size_t curve_index,
        const std::uint64_t analytical_branch_id,
        const std::string& analytical_branch_name)
    {
        // Provenance-aware archives are matched by stable branch identity,
        // independent of curve order or failed analytical branches.
        for(auto& curve: curve_container)
        {
            curve.set_main_refs(
                vec_ops, file_ops, log, nonlin_op, newton, cont_help);
            if(!curve.has_curve_provenance_metadata())
            {
                continue;
            }
            const auto& provenance = curve.get_curve_provenance();
            if(provenance.is_analytical() &&
               provenance.analytical_branch_id == analytical_branch_id)
            {
                return true;
            }
        }

        if(curve_index >= curve_container.size())
        {
            return false;
        }
        auto& curve = curve_container[curve_index];
        if(curve.has_curve_provenance_metadata())
        {
            return false;
        }

        curve.set_analytical_branch_provenance(
            analytical_branch_id,
            analytical_branch_name);
        log->warning_f(
            "container::bifurcation_diagram: restored missing analytical provenance for legacy curve %llu as exact branch %llu (%s).",
            static_cast<unsigned long long>(curve_index),
            static_cast<unsigned long long>(analytical_branch_id),
            analytical_branch_name.c_str());
        return true;
    }
    
    void init_new_curve()
    {
        write_legend_file();
        curve_number++;
        curve_container.emplace_back( vec_ops, file_ops, log, nonlin_op, newton, curve_number, directory, cont_help, skip_output ) ;
    }

    void pop_back_curve()
    {
        if(curve_number > 0)
        {
            curve_container.pop_back();
            curve_number--;
        }
    }

    void discard_current_curve()
    {
        if(curve_container.empty())
        {
            return;
        }
        curve_container.back().close_curve();
        curve_container.back().remove_output_directory();
        curve_container.pop_back();
        curve_number--;
    }

    void close_curve()
    {
        curve_container.back().close_curve();
    }

    bool commit_current_curve_symmetry_events()
    {
        if(curve_container.empty())
        {
            return true;
        }
        return curve_container.back().commit_staged_symmetry_events();
    }

    std::vector<symmetry_event_record_type> symmetry_event_records()
    {
        std::vector<symmetry_event_record_type> result;
        for(auto& curve: curve_container)
        {
            curve.set_main_refs(vec_ops, file_ops, log, nonlin_op, newton, cont_help);
            const auto& events = curve.symmetry_event_records();
            result.insert(result.end(), events.begin(), events.end());
        }
        return result;
    }

    std::vector<curve_point_type> get_curve_points_vector(int curve_number_)
    {
        try
        {
            auto &curve = curve_container.at(curve_number_);
            curve.set_main_refs(
                vec_ops,
                file_ops,
                log,
                nonlin_op,
                newton,
                cont_help);
            return( curve.return_curve_vector() );
        }
        catch(const std::exception& e)
        {
            log->warning_f("container::bifurcation_diagram::get_curve_points_vector: %s", e.what());
            std::vector<curve_point_type> zero;
            return(zero);
        }

    }

    bool read_saved_solution_from_curve(
        int curve_number_,
        const std::uint64_t source_point_index,
        T_vec& output)
    {
        if(
            curve_number_ < 0 ||
            static_cast<std::size_t>(curve_number_) >=
                curve_container.size())
        {
            return false;
        }
        auto& curve = curve_container[static_cast<std::size_t>(curve_number_)];
        curve.set_main_refs(
            vec_ops,
            file_ops,
            log,
            nonlin_op,
            newton,
            cont_help);
        return curve.read_saved_solution_at_source_index(
            source_point_index,
            output);
    }


    std::pair<bool, bool> get_solutoin_from_curve(int& curve_number_, int& container_index_, T& lambda_p, T_vec& x_p)
    {
        typename Curve::values_t point;
        bool metadata_available = false;
        return get_solutoin_from_curve(
            curve_number_,
            container_index_,
            lambda_p,
            x_p,
            point,
            metadata_available);
    }

    std::pair<bool, bool> get_solutoin_from_curve(
        int& curve_number_,
        int& container_index_,
        T& lambda_p,
        T_vec& x_p,
        typename Curve::values_t& point,
        bool& metadata_available)
    {
        if(curve_number_>curve_number)
        {
            log->error_f("requested curve number %i is not avaliable, current maximum number is %i.", curve_number_, curve_number);
            return std::make_pair(false, false);
        }
        else
        {
            auto &curve = curve_container.at(curve_number_);
            if(curve.is_curve_open())
            {
                log->error_f("requested curve number %i is opened and cannod be accessed unless it's closed", curve_number_);
                return std::make_pair(false, false);
            }
            curve.set_main_refs( vec_ops, file_ops, log, nonlin_op, newton, cont_help );
            bool is_there_a_solution = curve.get_avalible_solution(
                container_index_,
                lambda_p,
                x_p,
                point,
                metadata_available);
            if(is_there_a_solution)
            {
                return std::make_pair(true, true);
            }
            else
            {
                curve_number_++;
                return std::make_pair(true, false);
            }
        }
    }

    intersection_status find_intersection(const T& lambda_star, SolutionStorage*& solution_vector)
    {
        intersection_status status;
        for(auto &x: curve_container)
        {
            try
            {
                x.set_main_refs( vec_ops, file_ops, log, nonlin_op, newton, cont_help );
                status += x.find_intersection(lambda_star, solution_vector);
            }
            catch(const std::exception& e)
            {
                status.failed++;
                log->warning_f("container::bifurcation_diagram::find_intersection: %s", e.what());
            }
        }
        return status;

    }

    template<class SymmetryStorage>
    symmetry_archive_audit_result<T> audit_saved_symmetry_duplicates(
        SymmetryStorage* storage,
        const T parameter_tolerance,
        const double state_tolerance,
        const std::size_t minimum_matching_samples = 3)
    {
        if(storage == nullptr)
        {
            throw std::invalid_argument(
                "bifurcation_diagram symmetry audit requires storage");
        }
        if(!(parameter_tolerance >= T(0)) ||
           !(state_tolerance >= 0.0) ||
           minimum_matching_samples == 0)
        {
            throw std::invalid_argument(
                "bifurcation_diagram symmetry audit got invalid tolerances");
        }

        symmetry_archive_audit_result<T> result;
        T_vec left_state;
        T_vec right_state;
        vec_ops->init_vector(left_state);
        vec_ops->init_vector(right_state);
        vec_ops->start_use_vector(left_state);
        vec_ops->start_use_vector(right_state);
        const auto release = [this, &left_state, &right_state]()
        {
            vec_ops->stop_use_vector(right_state);
            vec_ops->free_vector(right_state);
            vec_ops->stop_use_vector(left_state);
            vec_ops->free_vector(left_state);
        };

        try
        {
            for(std::size_t first = 0;
                first < curve_container.size();
                ++first)
            {
                auto& first_curve = curve_container[first];
                first_curve.set_main_refs(
                    vec_ops, file_ops, log, nonlin_op, newton, cont_help);
                const auto first_points =
                    first_curve.return_curve_vector();
                for(std::size_t second = first + 1;
                    second < curve_container.size();
                    ++second)
                {
                    auto& second_curve = curve_container[second];
                    second_curve.set_main_refs(
                        vec_ops, file_ops, log, nonlin_op, newton, cont_help);
                    const auto second_points =
                        second_curve.return_curve_vector();

                    symmetry_duplicate_curve_pair<T> duplicate;
                    duplicate.first_curve = first;
                    duplicate.second_curve = second;
                    bool have_matching_sample = false;
                    for(const auto& left_point: first_points)
                    {
                        if(!left_point.is_data_avaliable)
                        {
                            continue;
                        }
                        bool left_loaded = false;
                        for(const auto& right_point: second_points)
                        {
                            if(!right_point.is_data_avaliable)
                            {
                                continue;
                            }
                            const T scale = T(1) + std::max(
                                std::abs(left_point.lambda),
                                std::abs(right_point.lambda));
                            if(std::abs(
                                   left_point.lambda -
                                   right_point.lambda) >
                               parameter_tolerance*scale)
                            {
                                continue;
                            }
                            if(!left_loaded)
                            {
                                if(!first_curve.read_saved_solution_for_audit(
                                       left_point,
                                       left_state))
                                {
                                    ++result.read_failures;
                                    break;
                                }
                                left_loaded = true;
                            }
                            if(!second_curve.read_saved_solution_for_audit(
                                   right_point,
                                   right_state))
                            {
                                ++result.read_failures;
                                continue;
                            }
                            ++result.compared_state_pairs;
                            const double distance =
                                storage->canonical_distance(
                                    left_state,
                                    right_state);
                            if(distance > state_tolerance)
                            {
                                continue;
                            }

                            const T parameter = T(0.5)*(
                                left_point.lambda +
                                right_point.lambda);
                            if(!have_matching_sample)
                            {
                                duplicate.minimum_parameter = parameter;
                                duplicate.maximum_parameter = parameter;
                                have_matching_sample = true;
                            }
                            else
                            {
                                duplicate.minimum_parameter = std::min(
                                    duplicate.minimum_parameter,
                                    parameter);
                                duplicate.maximum_parameter = std::max(
                                    duplicate.maximum_parameter,
                                    parameter);
                            }
                            ++duplicate.matching_samples;
                            duplicate.maximum_state_distance = std::max(
                                duplicate.maximum_state_distance,
                                distance);
                        }
                    }

                    const T minimum_span =
                        T(10)*parameter_tolerance;
                    if(duplicate.matching_samples >=
                           minimum_matching_samples &&
                       duplicate.maximum_parameter -
                           duplicate.minimum_parameter > minimum_span)
                    {
                        result.duplicate_curve_pairs.push_back(duplicate);
                    }
                }
            }
        }
        catch(...)
        {
            release();
            throw;
        }
        release();
        return result;
    }

    template<class StateDistance>
    bool find_branch_intersection(
        const T& step_lambda0,
        const T_vec& step_x0,
        const T& step_lambda1,
        const T_vec& step_x1,
        const branch_intersection_policy<T>& policy,
        T_vec& hit_x,
        branch_intersection_result<T>& result,
        StateDistance&& state_distance)
    {
        if(!policy.enabled)
        {
            return false;
        }

        std::vector<T> step_norms0;
        std::vector<T> step_norms1;
        nonlin_op->norm_bifurcation_diagram(step_x0, step_norms0);
        nonlin_op->norm_bifurcation_diagram(step_x1, step_norms1);

        for(auto &curve: curve_container)
        {
            curve.set_main_refs(vec_ops, file_ops, log, nonlin_op, newton, cont_help);
            if(curve.is_curve_open())
            {
                continue;
            }
            if(curve.find_branch_intersection(
                   step_lambda0,
                   step_x0,
                   step_lambda1,
                   step_x1,
                   step_norms0,
                   step_norms1,
                   policy,
                   hit_x,
                   result,
                   state_distance))
            {
                result.target_provenance = curve.get_curve_provenance();
                return true;
            }
        }
        return false;
    }



    void print_curves_status()
    {
        //log->info("container::bifurcation_diagram current curve number = %i\n", curve_number);
        std::cout << "container::bifurcation_diagram current curve number = " << curve_number << std::endl;
        for(auto &x: curve_container)
        {   
            x.print_curve_status();
        }  

    }

private:
    std::vector<Curve> curve_container;
    int curve_number = -1;

    void write_legend_file()
    {
        if(legend_written || nonlin_op == nullptr || directory.empty())
        {
            return;
        }

        const auto labels = bifurcation_diagram_detail::norm_labels(nonlin_op);
        if(labels.empty())
        {
            legend_written = true;
            return;
        }

        std::error_code ec;
        const std::filesystem::path project_dir(directory);
        if(!std::filesystem::is_directory(project_dir, ec))
        {
            return;
        }

        const auto legend_path = project_dir / "legend.dat";
        std::ofstream legend_file(legend_path);
        if(!legend_file)
        {
            if(log != nullptr)
            {
                log->warning_f("container::bifurcation_diagram: failed to open norm legend file: %s", legend_path.string().c_str());
            }
            return;
        }

        for(const auto& label: labels)
        {
            legend_file << label << '\n';
        }

        legend_written = true;
    }


    template<class Archive>
    void save(Archive& ar, const unsigned int) const
    {
        ar & curve_container;
        ar & skip_output;
        ar & curve_number;  //a curve number should be serialized!!!
        //ar & directory;      // should a directory be serialized?               
    }

    template<class Archive>
    void load(Archive& ar, const unsigned int)
    {
        ar & curve_container;
        ar & skip_output;
        ar & curve_number;

        // The project root is runtime configuration, not archive state.
        // Rebind every curve before any caller can resolve stored vectors or
        // sidecars through the path serialized by an older project location.
        if(cont_help != nullptr && !directory.empty())
        {
            reset_curve_output_directories();
        }
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER()

    
};





}

#endif // __BIFURCATION_DIAGRAM_H__
