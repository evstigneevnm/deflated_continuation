#ifndef __BIFURCATION_DIAGRAM_CURVE_H__
#define __BIFURCATION_DIAGRAM_CURVE_H__

#include <string>
#include <vector>
#include <stdexcept>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <filesystem>
#include <system_error>
#include <cstdint>
#include <sstream>
#include <algorithm>
#include <limits>


//using boost for serialization
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>

#include <containers/branch_intersection.h>
#include <containers/curve_endpoint_reason.h>
#include <containers/intersection_status.h>

namespace container
{

template<class T>
struct complex_values
{
    T lambda;
    bool is_data_avaliable = false;
    std::vector<T> vector_norms;
    uint64_t id_file_name;
    uint64_t point_index = 0;
    uint64_t segment_id = 0;
    uint64_t semicurve_id = 0;
    bool forced_store = false;
    curve_endpoint_reason endpoint_reason = curve_endpoint_reason::none;

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & lambda;
        // std::cout << "serialize: lambda = " << lambda << std::endl;
        ar & is_data_avaliable;
        // std::cout << "serialize: is_data_avaliable = " << is_data_avaliable << std::endl;
        ar & vector_norms;
        // for(auto &x: vector_norms)
        // {
        //     std::cout << "serialize: vector_norms = " << x << std::endl;
        // }
        ar & id_file_name;                        
        // std::cout << "serialize: id_file_name = " << id_file_name << std::endl;
    }

};

template<class VectorOperations, class VectorFileOperations, class Log, class NonlinearOperator, class Newton, class SolutionStorage, class HelperVectors>
class bifurcation_diagram_curve
{
private:
    friend class boost::serialization::access;

    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;

public:
    typedef complex_values<T> values_t;
    typedef std::vector<values_t> b_d_container_t;
    

    //for boost serialization!
    void set_main_refs(VectorOperations* vec_ops_, VectorFileOperations* vec_files_, Log* log_, NonlinearOperator* nlin_op_, Newton* newton_, HelperVectors* helper_vectors_)
    {
        if(!refs_set)
        {
            vec_ops = vec_ops_;
            vec_files = vec_files_;
            log = log_;
            nlin_op = nlin_op_;
            newton = newton_;
            helper_vectors_->get_refs(x0, x1);
            refs_set = true;
            load_metadata_if_available();
        }
    }

    bifurcation_diagram_curve(VectorOperations* vec_ops_, VectorFileOperations* vec_files_, Log* log_, NonlinearOperator* nlin_op_, Newton* newton_, int curve_number_, const std::string& directory_,  HelperVectors* helper_vectors_, unsigned int skip_output_):
    vec_ops(vec_ops_),
    vec_files(vec_files_),
    log(log_),
    nlin_op(nlin_op_),
    newton(newton_),
    skip_output(skip_output_)
    {
        refs_set = true;
        curve_open = true;
//        std::cout << "constructor of this " << this << " with "; //std::endl;
//NOTICE:
//it constructs these helper vectors every time we add this to the container
        //vec_ops->init_vector(x0); vec_ops->start_use_vector(x0);
        //vec_ops->init_vector(x1); vec_ops->start_use_vector(x1);
//too much ram will be used!
//a better solution is to use external vectors for this.
//but may couse logical problems?
//assume now that a HelperVectors class contains T_vec x0 and T_vec x1 and  can be accessed via reference.

        helper_vectors_->get_refs(x0, x1);
//        std::cout << "x0 = " << x0 << " x1 = " << x1 << std::endl;
        set_directory(directory_);
        set_curve_number(curve_number_);

        if(!debug_file.is_open())
            debug_file.open(debug_f_name.c_str(), std::ofstream::out | std::ofstream::app);

        log->info_f("container::bifurcation_diagram_curve(%i) opened.", curve_number);
    }
    
    bifurcation_diagram_curve()
    {
        //void default constructor for boost serialization
    }

    ~bifurcation_diagram_curve()
    {
//        std::cout << "distructor of this " << this << " with &x0 = " << x0 << " and &x1 = " << x1 << std::endl;
        //vec_ops->stop_use_vector(x0); vec_ops->free_vector(x0);
        //vec_ops->stop_use_vector(x1); vec_ops->free_vector(x1);
        if(debug_file.is_open())
            debug_file.close();
    }


    // bifurcation_diagram_curve(const bifurcation_diagram_curve& that)
    // {
    //     *this = that;
    //     std::cout << "copy constructor of " << this << std::endl;
    // }
    bifurcation_diagram_curve(const bifurcation_diagram_curve&) = delete;
    bifurcation_diagram_curve operator = (const bifurcation_diagram_curve&) = delete; //don't allow copy! Are we too fat?

    bifurcation_diagram_curve(bifurcation_diagram_curve&& that)
    {
//        std::cout << "move constructor of that " << &that << " to this " << this << std::endl;
        *this = std::move(that);
        
    }   

    bifurcation_diagram_curve& operator = (bifurcation_diagram_curve&& that)
    {
        if(&that == this)
        {
            return *this;
        }
        else
        {
//            std::cout << "move assign of that " << &that << " to this " << this << std::endl;
            vec_ops = that.vec_ops;
            vec_files = that.vec_files;
            log = that.log;
            nlin_op = that.nlin_op;
            newton = that.newton;
            data_directory = std::move(that.data_directory);
            full_path = std::move(that.full_path);
            container = std::move(that.container);
            global_id = that.global_id;
            global_index = that.global_index;
            curve_number = that.curve_number;
            x0 = that.x0;
            x1 = that.x1;
            lambda0 = that.lambda0;
            lambda1 = that.lambda1;
            skip_output = that.skip_output;
            debug_f_name  = std::move(that.debug_f_name);
            metadata_f_name = std::move(that.metadata_f_name);
            debug_file = std::move(that.debug_file); //std::move(that.debug_file). Move of std::ofstream supported only from C++5.X and above!
            curve_open = that.curve_open;
            refs_set = that.refs_set;
            current_segment_id = that.current_segment_id;
            current_semicurve_id = that.current_semicurve_id;
            segment_metadata_available = that.segment_metadata_available;
            incomplete_segment_ids = std::move(that.incomplete_segment_ids);
            return *this;
            
            //
            // Judas Priest \m/
            // One shot at glory? =)
            //
        }
    }


private:
    VectorOperations* vec_ops;
    VectorFileOperations* vec_files;
    Log* log;
    NonlinearOperator* nlin_op;
    Newton* newton;
    std::string data_directory = "dat_files";
    int curve_number = 0;
    std::string full_path = ".";
    unsigned int skip_output;
    std::string debug_f_name;
    std::string metadata_f_name;
    std::ofstream debug_file;
    bool refs_set = false;
    uint64_t current_segment_id = 0;
    uint64_t current_semicurve_id = 0;
    bool segment_metadata_available = false;
    std::vector<uint64_t> incomplete_segment_ids;

    void mark_incomplete_segment(const uint64_t segment_id)
    {
        if(std::find(incomplete_segment_ids.begin(), incomplete_segment_ids.end(), segment_id) == incomplete_segment_ids.end())
        {
            incomplete_segment_ids.push_back(segment_id);
        }
    }

    bool is_incomplete_segment(const uint64_t segment_id) const
    {
        return std::find(incomplete_segment_ids.begin(), incomplete_segment_ids.end(), segment_id) != incomplete_segment_ids.end();
    }


    bool directory_exists(const std::string& name) const
    {
        std::error_code ec;
        return std::filesystem::is_directory(name, ec);
    }

    bool path_exists(const std::string& name) const
    {
        std::error_code ec;
        return std::filesystem::exists(name, ec);
    }

    void ensure_curve_directory_exists()
    {
        if(directory_exists(full_path))
        {
            return;
        }

        if(path_exists(full_path))
        {
            throw std::runtime_error(
                std::string("container::bifurcation_diagram_curve: curve output path exists but is not a directory: ") +
                full_path);
        }

        std::error_code ec;
        if(!std::filesystem::create_directory(full_path, ec))
        {
            if(ec)
            {
                throw std::runtime_error(
                    std::string("container::bifurcation_diagram_curve: failed to create curve output directory '") +
                    full_path + "': " + ec.message());
            }
            if(!directory_exists(full_path))
            {
                throw std::runtime_error(
                    std::string("container::bifurcation_diagram_curve: failed to create curve output directory: ") +
                    full_path);
            }
        }
        log->info_f("container::bifurcation_diagram_curve: created curve output directory: %s", full_path.c_str());
    }

    void write_metadata_file()
    {
        if(metadata_f_name.empty())
        {
            return;
        }
        std::ofstream f(metadata_f_name.c_str(), std::ofstream::out);
        if(!f)
        {
            if(log != nullptr)
            {
                log->warning_f("container::bifurcation_diagram_curve(%i): failed to open metadata file %s", curve_number, metadata_f_name.c_str());
            }
            return;
        }
        f << "# index lambda saved id_file_name segment_id semicurve_id forced_store endpoint_reason\n";
        for(std::size_t i = 0; i < container.size(); ++i)
        {
            const auto& x = container[i];
            f << i << " "
              << std::setprecision(16) << x.lambda << " "
              << (x.is_data_avaliable ? 1 : 0) << " "
              << x.id_file_name << " "
              << x.segment_id << " "
              << x.semicurve_id << " "
              << (x.forced_store ? 1 : 0) << " "
              << to_string(x.endpoint_reason) << "\n";
        }
    }

    void load_metadata_if_available()
    {
        if(metadata_f_name.empty())
        {
            metadata_f_name.assign(full_path + std::string("/") + std::string("metadata_curve.dat"));
        }
        std::ifstream f(metadata_f_name.c_str());
        if(!f)
        {
            segment_metadata_available = false;
            return;
        }

        std::string line;
        bool loaded_any = false;
        incomplete_segment_ids.clear();
        while(std::getline(f, line))
        {
            if(line.empty() || line[0] == '#')
            {
                continue;
            }
            std::istringstream stream(line);
            std::size_t index = 0;
            T lambda = T(0);
            unsigned int saved = 0;
            uint64_t id_file_name = 0;
            uint64_t segment_id = 0;
            uint64_t semicurve_id = 0;
            unsigned int forced_store = 0;
            std::string endpoint_reason_value;
            stream >> index >> lambda >> saved >> id_file_name >> segment_id >> semicurve_id >> forced_store;
            if(!stream || index >= container.size())
            {
                continue;
            }
            stream >> endpoint_reason_value;
            auto& point = container[index];
            point.point_index = static_cast<uint64_t>(index);
            point.segment_id = segment_id;
            point.semicurve_id = semicurve_id;
            point.forced_store = forced_store != 0;
            point.endpoint_reason = curve_endpoint_reason_from_string(endpoint_reason_value);
            if(is_incomplete_endpoint(point.endpoint_reason))
            {
                mark_incomplete_segment(segment_id);
            }
            loaded_any = true;
        }
        segment_metadata_available = loaded_any;
    }

    bool can_interpolate_between(const values_t& lower, const values_t& upper) const
    {
        if(!segment_metadata_available)
        {
            return true;
        }
        return lower.segment_id == upper.segment_id && !is_incomplete_segment(lower.segment_id);
    }

    bool is_incomplete_pair(const values_t& lower, const values_t& upper) const
    {
        return segment_metadata_available &&
               lower.segment_id == upper.segment_id &&
               is_incomplete_segment(lower.segment_id);
    }

    bool point_in_segment(const values_t& point, const uint64_t segment_id) const
    {
        return !segment_metadata_available || point.segment_id == segment_id;
    }

    static T scalar_abs_value(const T& value)
    {
        return value < T(0) ? -value : value;
    }

    static bool same_scalar(const T& a, const T& b)
    {
        const T scale = std::max<T>(T(1), std::max<T>(scalar_abs_value(a), scalar_abs_value(b)));
        return scalar_abs_value(a - b) <= T(64)*std::numeric_limits<T>::epsilon()*scale;
    }

    static bool interval_overlap(
        const T& a0,
        const T& a1,
        const T& b0,
        const T& b1,
        T& lower,
        T& upper)
    {
        const T a_min = std::min(a0, a1);
        const T a_max = std::max(a0, a1);
        const T b_min = std::min(b0, b1);
        const T b_max = std::max(b0, b1);
        lower = std::max(a_min, b_min);
        upper = std::min(a_max, b_max);
        return lower <= upper || same_scalar(lower, upper);
    }

    static T interpolation_weight(const T& lambda, const T& lambda0_, const T& lambda1_)
    {
        if(same_scalar(lambda0_, lambda1_))
        {
            return T(0.5);
        }
        return (lambda - lambda0_)/(lambda1_ - lambda0_);
    }

    static T interpolate_scalar(
        const T& lambda,
        const T& lambda0_,
        const T& value0,
        const T& lambda1_,
        const T& value1)
    {
        const T w = interpolation_weight(lambda, lambda0_, lambda1_);
        return (T(1) - w)*value0 + w*value1;
    }

    static bool get_signature_value(
        const values_t& point,
        const unsigned int signature_index,
        T& value)
    {
        if(point.vector_norms.size() <= signature_index)
        {
            return false;
        }
        value = point.vector_norms[signature_index];
        return true;
    }

    static bool signature_envelopes_overlap(
        const T& new_signature0,
        const T& new_signature1,
        const T& old_signature0,
        const T& old_signature1,
        const T& tolerance)
    {
        const T new_min = std::min(new_signature0, new_signature1);
        const T new_max = std::max(new_signature0, new_signature1);
        const T old_min = std::min(old_signature0, old_signature1);
        const T old_max = std::max(old_signature0, old_signature1);
        return (new_min - tolerance) <= old_max && (old_min - tolerance) <= new_max;
    }

    static bool find_signature_candidate_lambda(
        const T& lambda_lower,
        const T& lambda_upper,
        const T& step_lambda0,
        const T& step_signature0,
        const T& step_lambda1,
        const T& step_signature1,
        const T& old_lambda0,
        const T& old_signature0,
        const T& old_lambda1,
        const T& old_signature1,
        const T& tolerance,
        T& candidate_lambda,
        T& signature_distance)
    {
        const T d_lower =
            interpolate_scalar(lambda_lower, step_lambda0, step_signature0, step_lambda1, step_signature1) -
            interpolate_scalar(lambda_lower, old_lambda0, old_signature0, old_lambda1, old_signature1);
        const T d_upper =
            interpolate_scalar(lambda_upper, step_lambda0, step_signature0, step_lambda1, step_signature1) -
            interpolate_scalar(lambda_upper, old_lambda0, old_signature0, old_lambda1, old_signature1);

        const T abs_lower = scalar_abs_value(d_lower);
        const T abs_upper = scalar_abs_value(d_upper);
        if(abs_lower <= tolerance || same_scalar(lambda_lower, lambda_upper))
        {
            candidate_lambda = lambda_lower;
            signature_distance = abs_lower;
            return signature_distance <= tolerance;
        }
        if(abs_upper <= tolerance)
        {
            candidate_lambda = lambda_upper;
            signature_distance = abs_upper;
            return true;
        }
        if(d_lower*d_upper > T(0))
        {
            return false;
        }

        const T denominator = d_upper - d_lower;
        if(same_scalar(denominator, T(0)))
        {
            candidate_lambda = T(0.5)*(lambda_lower + lambda_upper);
        }
        else
        {
            candidate_lambda = lambda_lower - d_lower*(lambda_upper - lambda_lower)/denominator;
        }
        if(candidate_lambda < lambda_lower || candidate_lambda > lambda_upper)
        {
            return false;
        }
        const T d_candidate =
            interpolate_scalar(candidate_lambda, step_lambda0, step_signature0, step_lambda1, step_signature1) -
            interpolate_scalar(candidate_lambda, old_lambda0, old_signature0, old_lambda1, old_signature1);
        signature_distance = scalar_abs_value(d_candidate);
        return signature_distance <= tolerance;
    }

    static bool candidate_has_step_progress(
        const T& candidate_lambda,
        const T& step_lambda0,
        const T& step_lambda1,
        const T& minimum_step_fraction_from_start)
    {
        if(same_scalar(step_lambda0, step_lambda1))
        {
            return false;
        }
        const T fraction = (candidate_lambda - step_lambda0)/(step_lambda1 - step_lambda0);
        return fraction > minimum_step_fraction_from_start && fraction <= T(1) + minimum_step_fraction_from_start;
    }

public:



    void set_directory(const std::string& data_directory_)
    {
        data_directory.assign(data_directory_);
        if(!directory_exists(data_directory))
        {
            if(path_exists(data_directory))
            {
                throw std::runtime_error(
                    std::string("container::bifurcation_diagram_curve: project path exists but is not a directory: ") +
                    data_directory);
            }
            throw std::runtime_error(
                std::string("container::bifurcation_diagram_curve: project directory doesn't exist: ") +
                data_directory);
        }
    }


    void set_curve_number(int curve_number_)
    {
        curve_number = curve_number_;
        full_path.assign((std::filesystem::path(data_directory) / std::to_string(curve_number)).string());
        debug_f_name.assign(full_path.c_str() + std::string("/") + std::string("debug_curve.dat"));
        metadata_f_name.assign(full_path.c_str() + std::string("/") + std::string("metadata_curve.dat"));
        log->info_f("container::bifurcation_diagram_curve: FULL PATH: %s", full_path.c_str());
        ensure_curve_directory_exists();

    }

    void reset_output_directory(const std::string& data_directory_)
    {
        set_directory(data_directory_);
        set_curve_number(curve_number);
        load_metadata_if_available();
    }

    void start_new_segment()
    {
        current_segment_id++;
        current_semicurve_id = current_segment_id;
        segment_metadata_available = true;
    }

    void add(
        const T& lambda_,
        const T_vec& x_,
        bool force_store = false,
        curve_endpoint_reason endpoint_reason = curve_endpoint_reason::none)
    {
        if(curve_open)
        {
            std::vector<T> bif_diag_norms;
            nlin_op->norm_bifurcation_diagram(x_, bif_diag_norms);
            store_t store_result = store(lambda_, x_, force_store);

            values_t form_values;
            form_values.lambda = lambda_;
            form_values.is_data_avaliable = store_result.first;
            form_values.id_file_name = store_result.second;
            form_values.vector_norms = bif_diag_norms;
            form_values.point_index = static_cast<uint64_t>(container.size());
            form_values.segment_id = current_segment_id;
            form_values.semicurve_id = current_semicurve_id;
            form_values.forced_store = force_store;
            form_values.endpoint_reason = endpoint_reason;
            if(is_incomplete_endpoint(endpoint_reason))
            {
                mark_incomplete_segment(form_values.segment_id);
            }

            container.push_back(form_values);

            print_curve_each();
        }
        else
        {
            throw std::runtime_error(std::string("container::bifurcation_diagram_curve: trying to add to a closed curve!") );
        }    
    }

    std::string get_full_path()
    {
        return(full_path);
    }

    void set_last_endpoint_reason(curve_endpoint_reason endpoint_reason)
    {
        if(container.empty())
        {
            return;
        }
        container.back().endpoint_reason = endpoint_reason;
        if(is_incomplete_endpoint(endpoint_reason))
        {
            mark_incomplete_segment(container.back().segment_id);
        }
        write_metadata_file();
    }

    //return a solution pair (x,\lambda) from the container
    //for the stability analysis. 
    //Should be done as a querry operation.
    //container_index is returned to the upper level
    //returned 'true' means that the pair is found, 'false' - that there are no more pairs
    bool get_avalible_solution(int& container_index, T& lambda_p, T_vec& x_p)
    {
        int N = container.size();
        if(container_index<N)
        {
            bool solution_found = false;
            int j = 0;
            for(j=container_index;j<N;j++)
            {
                auto &p_j = container[j];
                if(p_j.is_data_avaliable)
                {
                    uint64_t local_id = p_j.id_file_name;
                    std::string f_name = full_path+std::string("/")+std::to_string(local_id);
                    vec_files->read_vector(f_name, x_p);
                    lambda_p = p_j.lambda;
                    solution_found = true;
                    log->info_f("bifurcation_diagram_curve::get_avalible_solution: got solution from %s", f_name.c_str());
                    break;
                }
            }
            container_index = j+1;
            if(solution_found)
                return true;
            else
                return false;
        }
        else
        {
            return false;
        }
    }

    //intersect solutions
    intersection_status find_intersection(const T& lambda_star, SolutionStorage*& solution_vector)
    {
        intersection_status status;
        int N = container.size();
        for(int j=0;j<N-1;j++)
        {
            int ind = j;
            int indp = j+1;
            auto &p_j = container[ind];
            auto &p_jp = container[indp];
            if(!intersection(p_j, p_jp, lambda_star))
            {
                continue;
            }
            if(!can_interpolate_between(p_j, p_jp))
            {
                if(is_incomplete_pair(p_j, p_jp))
                {
                    status.skipped_incomplete++;
                }
                else
                {
                    status.skipped_discontinuous++;
                }
                continue;
            }
            {
                if((p_j.lambda == lambda_star)&&(p_j.is_data_avaliable))
                {
                    uint64_t local_id = p_j.id_file_name;
                    std::string f_name = full_path+std::string("/")+std::to_string(local_id);
                    vec_files->read_vector(f_name, x1); 
                    solution_vector->push_back(x1); 
                    status.added++;
                    log->info_f("container::bifurcation_diagram_curve(%i): added intersectoin at (%i) for the solution at lambda =  %lf", curve_number, ind, lambda_star);               
                }
                else if((p_jp.lambda == lambda_star)&&(p_jp.is_data_avaliable))
                {
                    uint64_t local_id = p_jp.id_file_name;
                    std::string f_name = full_path+std::string("/")+std::to_string(local_id);
                    vec_files->read_vector(f_name, x1); 
                    solution_vector->push_back(x1); 
                    status.added++;
                    log->info_f("container::bifurcation_diagram_curve(%i): added intersectoin at (%i) for the solution at lambda =  %lf", curve_number, indp, lambda_star);               

                }
                else
                {
                    bool stat_l = get_lower(ind, p_j.segment_id);
                    bool stat_u = get_upper(indp, p_jp.segment_id);
                    if(stat_l&&stat_u)
                    {
                        if(interpolate_solutions(lambda_star))
                        {
                            solution_vector->push_back(x1);
                            status.added++;
                            log->info_f("container::bifurcation_diagram_curve(%i): added intersectoin at (%i,%i) for the solution at lambda =  %lf", curve_number, ind, indp, lambda_star);
                        }
                        else
                        {
                            //throw std::runtime_error(std::string("container::bifurcation_diagram_curve: newton failed in solution section") );
                            status.failed++;
                            log->warning_f("container::bifurcation_diagram_curve(%i): !!!newton failed in solution section at (%i(%i), %i(%i)) for the solution at lambda =  %lf !!!", curve_number, ind, int(stat_l), indp, int(stat_u), lambda_star);
                        }
                    }
                    else
                    {
                        //std::string fail_find_files = std::string("container::bifurcation_diagram_curve(") + std::to_string(curve_number) + std::string("): failed to find a valid solution for the parameter = ") + std::to_string(lambda_star) + std::string(" with lower flag = ") + std::to_string(stat_l) + std::string(" and upper flag = ") + std::to_string(stat_u) + std::string(", indexing = (") + std::to_string(ind) + std::string(",") + std::to_string(indp) + std::string(").");
                        status.missing_data++;
                        log->warning_f("container::bifurcation_diagram_curve(%i): !!!failed to add intersectoin at (%i(%i), %i(%i)) for the solution at lambda =  %lf !!!", curve_number, ind, int(stat_l), indp, int(stat_u), lambda_star);

                        //throw std::runtime_error( fail_find_files );
                    }
                }
            }
        }
        return status;

    }

    bool evaluate_at_lambda(const T& lambda_star, T_vec& x_out)
    {
        const int N = static_cast<int>(container.size());
        for(int j = 0; j < N - 1; ++j)
        {
            const auto& p_j = container[j];
            const auto& p_jp = container[j + 1];
            if(!can_interpolate_between(p_j, p_jp))
            {
                continue;
            }
            if(intersection(p_j, p_jp, lambda_star))
            {
                return evaluate_segment_at_lambda(j, j + 1, lambda_star, x_out);
            }
        }
        return false;
    }

    template<class StateDistance>
    bool find_branch_intersection(
        const T& step_lambda0,
        const T_vec& step_x0,
        const T& step_lambda1,
        const T_vec& step_x1,
        const std::vector<T>& step_norms0,
        const std::vector<T>& step_norms1,
        const branch_intersection_policy<T>& policy,
        T_vec& hit_x,
        branch_intersection_result<T>& result,
        StateDistance&& state_distance)
    {
        if(!policy.enabled)
        {
            return false;
        }
        if(step_norms0.size() <= policy.signature_norm_index ||
           step_norms1.size() <= policy.signature_norm_index)
        {
            return false;
        }

        const T step_signature0 = step_norms0[policy.signature_norm_index];
        const T step_signature1 = step_norms1[policy.signature_norm_index];
        const int N = static_cast<int>(container.size());
        for(int j = 0; j < N - 1; ++j)
        {
            const auto& p_j = container[j];
            const auto& p_jp = container[j + 1];
            if(!can_interpolate_between(p_j, p_jp))
            {
                continue;
            }

            T lambda_lower = T(0);
            T lambda_upper = T(0);
            if(!interval_overlap(step_lambda0, step_lambda1, p_j.lambda, p_jp.lambda, lambda_lower, lambda_upper))
            {
                continue;
            }

            T old_signature0 = T(0);
            T old_signature1 = T(0);
            if(!get_signature_value(p_j, policy.signature_norm_index, old_signature0) ||
               !get_signature_value(p_jp, policy.signature_norm_index, old_signature1))
            {
                continue;
            }

            const T signature_scale = std::max<T>(
                T(1),
                std::max<T>(
                    std::max<T>(scalar_abs_value(step_signature0), scalar_abs_value(step_signature1)),
                    std::max<T>(scalar_abs_value(old_signature0), scalar_abs_value(old_signature1))));
            const T signature_tolerance = policy.signature_tolerance*signature_scale;
            if(!signature_envelopes_overlap(
                   step_signature0,
                   step_signature1,
                   old_signature0,
                   old_signature1,
                   signature_tolerance))
            {
                continue;
            }

            T candidate_lambda = T(0);
            T signature_distance = T(0);
            if(!find_signature_candidate_lambda(
                   lambda_lower,
                   lambda_upper,
                   step_lambda0,
                   step_signature0,
                   step_lambda1,
                   step_signature1,
                   p_j.lambda,
                   old_signature0,
                   p_jp.lambda,
                   old_signature1,
                   signature_tolerance,
                   candidate_lambda,
                   signature_distance))
            {
                continue;
            }
            if(!candidate_has_step_progress(
                   candidate_lambda,
                   step_lambda0,
                   step_lambda1,
                   policy.minimum_step_fraction_from_start))
            {
                candidate_lambda = lambda_upper;
                const T d_upper =
                    interpolate_scalar(
                        candidate_lambda,
                        step_lambda0,
                        step_signature0,
                        step_lambda1,
                        step_signature1) -
                    interpolate_scalar(
                        candidate_lambda,
                        p_j.lambda,
                        old_signature0,
                        p_jp.lambda,
                        old_signature1);
                signature_distance = scalar_abs_value(d_upper);
                if(signature_distance > signature_tolerance ||
                   !candidate_has_step_progress(
                       candidate_lambda,
                       step_lambda0,
                       step_lambda1,
                       policy.minimum_step_fraction_from_start))
                {
                    candidate_lambda = T(0.5)*(lambda_lower + lambda_upper);
                    const T d_mid =
                        interpolate_scalar(
                            candidate_lambda,
                            step_lambda0,
                            step_signature0,
                            step_lambda1,
                            step_signature1) -
                        interpolate_scalar(
                            candidate_lambda,
                            p_j.lambda,
                            old_signature0,
                            p_jp.lambda,
                            old_signature1);
                    signature_distance = scalar_abs_value(d_mid);
                    if(signature_distance > signature_tolerance ||
                       !candidate_has_step_progress(
                           candidate_lambda,
                           step_lambda0,
                           step_lambda1,
                           policy.minimum_step_fraction_from_start))
                    {
                        continue;
                    }
                }
            }

            if(!evaluate_segment_at_lambda(j, j + 1, candidate_lambda, x1))
            {
                continue;
            }

            const T w = interpolation_weight(candidate_lambda, step_lambda0, step_lambda1);
            vec_ops->assign_mul(T(1) - w, step_x0, w, step_x1, x0);
            const T distance = static_cast<T>(state_distance(x0, x1));
            const T state_scale = std::max<T>(
                T(1),
                std::max<T>(
                    scalar_abs_value(interpolate_scalar(
                        candidate_lambda,
                        step_lambda0,
                        step_signature0,
                        step_lambda1,
                        step_signature1)),
                    scalar_abs_value(interpolate_scalar(
                        candidate_lambda,
                        p_j.lambda,
                        old_signature0,
                        p_jp.lambda,
                        old_signature1))));
            const T state_tolerance = policy.state_tolerance*state_scale;
            if(distance <= state_tolerance)
            {
                vec_ops->assign(x1, hit_x);
                result.found = true;
                result.lambda = candidate_lambda;
                result.signature_distance = signature_distance;
                result.state_distance = distance;
                result.state_tolerance = state_tolerance;
                result.curve_number = curve_number;
                result.segment_id = p_j.segment_id;
                result.semicurve_id = p_j.semicurve_id;
                result.lower_point_index = p_j.point_index;
                result.upper_point_index = p_jp.point_index;
                result.reason = "known_branch_intersection";
                return true;
            }
        }
        return false;
    }

    template<class StateDistance>
    bool find_self_intersection(
        const T& step_lambda0,
        const T_vec& step_x0,
        const T& step_lambda1,
        const T_vec& step_x1,
        const std::vector<T>& step_norms0,
        const std::vector<T>& step_norms1,
        const self_intersection_policy<T>& policy,
        T_vec& hit_x,
        branch_intersection_result<T>& result,
        StateDistance&& state_distance)
    {
        if(!policy.enabled)
        {
            return false;
        }
        if(step_norms0.size() <= policy.signature_norm_index ||
           step_norms1.size() <= policy.signature_norm_index ||
           container.size() < 2)
        {
            return false;
        }

        const uint64_t latest_index = container.back().point_index;
        const T step_signature0 = step_norms0[policy.signature_norm_index];
        const T step_signature1 = step_norms1[policy.signature_norm_index];
        const int N = static_cast<int>(container.size());
        for(int j = 0; j < N - 1; ++j)
        {
            const auto& p_j = container[j];
            const auto& p_jp = container[j + 1];
            const uint64_t candidate_latest_index = std::max(p_j.point_index, p_jp.point_index);
            if(latest_index <= candidate_latest_index + policy.minimum_index_gap)
            {
                continue;
            }
            if(!can_interpolate_between(p_j, p_jp))
            {
                continue;
            }

            T lambda_lower = T(0);
            T lambda_upper = T(0);
            if(!interval_overlap(step_lambda0, step_lambda1, p_j.lambda, p_jp.lambda, lambda_lower, lambda_upper))
            {
                continue;
            }

            T old_signature0 = T(0);
            T old_signature1 = T(0);
            if(!get_signature_value(p_j, policy.signature_norm_index, old_signature0) ||
               !get_signature_value(p_jp, policy.signature_norm_index, old_signature1))
            {
                continue;
            }

            const T signature_scale = std::max<T>(
                T(1),
                std::max<T>(
                    std::max<T>(scalar_abs_value(step_signature0), scalar_abs_value(step_signature1)),
                    std::max<T>(scalar_abs_value(old_signature0), scalar_abs_value(old_signature1))));
            const T signature_tolerance = policy.signature_tolerance*signature_scale;
            if(!signature_envelopes_overlap(
                   step_signature0,
                   step_signature1,
                   old_signature0,
                   old_signature1,
                   signature_tolerance))
            {
                continue;
            }

            T candidate_lambda = T(0);
            T signature_distance = T(0);
            if(!find_signature_candidate_lambda(
                   lambda_lower,
                   lambda_upper,
                   step_lambda0,
                   step_signature0,
                   step_lambda1,
                   step_signature1,
                   p_j.lambda,
                   old_signature0,
                   p_jp.lambda,
                   old_signature1,
                   signature_tolerance,
                   candidate_lambda,
                   signature_distance))
            {
                continue;
            }
            if(!candidate_has_step_progress(
                   candidate_lambda,
                   step_lambda0,
                   step_lambda1,
                   policy.minimum_step_fraction_from_start))
            {
                continue;
            }

            if(!evaluate_segment_at_lambda(j, j + 1, candidate_lambda, x1))
            {
                continue;
            }

            const T w = interpolation_weight(candidate_lambda, step_lambda0, step_lambda1);
            vec_ops->assign_mul(T(1) - w, step_x0, w, step_x1, x0);
            const T distance = static_cast<T>(state_distance(x0, x1));
            const T state_scale = std::max<T>(
                T(1),
                std::max<T>(
                    scalar_abs_value(interpolate_scalar(
                        candidate_lambda,
                        step_lambda0,
                        step_signature0,
                        step_lambda1,
                        step_signature1)),
                    scalar_abs_value(interpolate_scalar(
                        candidate_lambda,
                        p_j.lambda,
                        old_signature0,
                        p_jp.lambda,
                        old_signature1))));
            const T state_tolerance = policy.state_tolerance*state_scale;
            if(distance <= state_tolerance)
            {
                vec_ops->assign(x1, hit_x);
                result.found = true;
                result.lambda = candidate_lambda;
                result.signature_distance = signature_distance;
                result.state_distance = distance;
                result.state_tolerance = state_tolerance;
                result.curve_number = curve_number;
                result.segment_id = p_j.segment_id;
                result.semicurve_id = p_j.semicurve_id;
                result.lower_point_index = p_j.point_index;
                result.upper_point_index = p_jp.point_index;
                result.reason = "self_intersection";
                return true;
            }
        }
        return false;
    }






// Debug print out for gnuplot
    void print_curve_each()
    {
        size_t container_size = container.size();
        if(container_size%(skip_output) == 0)
        {
            b_d_container_t local_container(container.end()-skip_output, container.end());

            for(auto &x: local_container)
            {
                debug_file << std::setprecision(16) << x.lambda << " ";
                for(auto &y: x.vector_norms)
                {
                    debug_file << std::setprecision(16) << y << " ";  //print all avaliable norms!
                }
                debug_file << x.id_file_name << std::endl;
            }
            debug_file.flush();
            log->info_f("container::bifurcation_diagram_curve(%i): printed debug bifurcation curve data.", curve_number); 
        }
    }

    void print_curve()
    {
        std::string f_name = full_path + std::string("/") + std::string("debug_curve_all.dat");
        std::ofstream f(f_name.c_str(), std::ofstream::out);
        for(auto &x: container)
        {
            f << std::setprecision(16) << x.lambda << " ";
            for(auto &y: x.vector_norms)
            {
                f << std::setprecision(16) << y << " ";  //print all avaliable norms!
            }
            f << x.id_file_name << std::endl;
        }
        f.close();
        write_metadata_file();
        log->info_f("container::bifurcation_diagram_curve(%i): printed final bifurcation curve data.", curve_number); 
    }


    void print_curve_status()
    {
        std::cout << "container::bifurcation_diagram_curve::curve number = " << curve_number << std::endl;
        std::cout << "container::bifurcation_diagram_curve::container_length = " << container.size() << std::endl;
    }

    void close_curve()
    {
        
        container.shrink_to_fit();
        if(debug_file.is_open())
            debug_file.close();
        write_metadata_file();
        curve_open = false; 
        log->info_f("container::bifurcation_diagram_curve(%i) closed.", curve_number); 
    }

    void remove_output_directory()
    {
        if(debug_file.is_open())
        {
            debug_file.close();
        }
        std::error_code ec;
        std::filesystem::remove_all(full_path, ec);
        if(ec)
        {
            log->warning_f(
                "container::bifurcation_diagram_curve(%i): failed to remove discarded curve directory %s: %s",
                curve_number,
                full_path.c_str(),
                ec.message().c_str());
        }
        else
        {
            log->info_f(
                "container::bifurcation_diagram_curve(%i): removed discarded curve directory %s",
                curve_number,
                full_path.c_str());
        }
    }


    bool is_curve_open()
    {
        return(curve_open);
    }


//TODO: function that adds to a specific file at every step for monitoring.
    

//TODO: add delete function that removes all information from the container for given interval of lambdas
    void delete_ponts(const int& lambda_min, const int& lambda_max)
    {


    }

private:
    typedef std::pair<bool, uint64_t> store_t;
public:
    //takes some memory, can be used only for visualization
    //makes a copy so that original container is undamaged!
    b_d_container_t return_curve_vector()
    {
        return(container);
    }

private:
    b_d_container_t container;
    uint64_t global_id = 0; 
    uint64_t global_index = 0;

    T_vec x0;
    T_vec x1;
    T lambda0, lambda1;
    bool curve_open;


    //boost serialization
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & data_directory;
        ar & curve_number;
        ar & full_path;     // do we need it???
        ar & skip_output;   
        ar & debug_f_name; 
        ar & container;
        ar & global_id;
        ar & global_index;
        ar & curve_open;

        //should we add x0 and x1? we won't be able to continue the curve unless these are added
    }

    store_t store(const T& lambda_, const T_vec& x_, bool force_store_)
    {
        //TODO: add condition for storing data on the drive
        store_t res;
        
        if( ((global_index++)%skip_output==0)||(force_store_) )
        {
            res.first = true;
        }
        else
        {
            res.first = false;
        }
        

        if(res.first)
        {
            global_id++;
            std::string f_name = full_path.c_str()+std::string("/")+std::to_string(global_id);
            log->info_f("container::bifurcation_diagram_curve: FULL PATH: %s", full_path.c_str());
            vec_files->write_vector(f_name, x_);
            res.second = global_id;
        }
        else
        {
            res.second = 0;
        }
        return(res);
    }

    bool read_saved_point(const values_t& point, T_vec& x_out)
    {
        if(!point.is_data_avaliable)
        {
            return false;
        }
        const uint64_t local_id = point.id_file_name;
        const std::string f_name = full_path + std::string("/") + std::to_string(local_id);
        vec_files->read_vector(f_name, x_out);
        return true;
    }

    bool evaluate_segment_at_lambda(const int lower_index, const int upper_index, const T& lambda_star, T_vec& x_out)
    {
        auto& p_j = container[lower_index];
        auto& p_jp = container[upper_index];
        if((same_scalar(p_j.lambda, lambda_star)) && read_saved_point(p_j, x_out))
        {
            return true;
        }
        if((same_scalar(p_jp.lambda, lambda_star)) && read_saved_point(p_jp, x_out))
        {
            return true;
        }

        const bool stat_l = get_lower(lower_index, p_j.segment_id);
        const bool stat_u = get_upper(upper_index, p_jp.segment_id);
        if(!stat_l || !stat_u)
        {
            return false;
        }
        if(!interpolate_solutions(lambda_star))
        {
            return false;
        }
        vec_ops->assign(x1, x_out);
        return true;
    }

    bool get_lower(int index, uint64_t segment_id)
    {

        int j = index;
        bool saved_data = false;
        while(!saved_data)
        {
            values_t local_data = container[j];
            if(!point_in_segment(local_data, segment_id))
            {
                break;
            }
            saved_data = local_data.is_data_avaliable;
            if(saved_data)
            {
                lambda0 = local_data.lambda;
                uint64_t local_id = local_data.id_file_name;
                std::string f_name = full_path.c_str()+std::string("/")+std::to_string(local_id);
                vec_files->read_vector(f_name, x0);
                break;
            }
            j--;
            if(j<0)
                break;


        }
        return(saved_data);
        
    }
    bool get_upper(int index, uint64_t segment_id)
    {

        int j = index;
        bool saved_data = false;
        while(!saved_data)
        {
            int container_size = container.size();
            //std::cout << "container_size = " << container_size << std::endl;
            
            values_t local_data = container[j];
            if(!point_in_segment(local_data, segment_id))
            {
                break;
            }
            saved_data = local_data.is_data_avaliable;
            if(saved_data)
            {
                lambda1 = local_data.lambda;
                uint64_t local_id = local_data.id_file_name;
                std::string f_name = full_path+std::string("/")+std::to_string(local_id);
                vec_files->read_vector(f_name, x1);
                break;
            }
            j++;
            if( j >= container_size )
                break;
        }
        return(saved_data);
        
    }
    bool intersection(const values_t& x_, const values_t& xp_, const T& lambda_)
    {
        if((x_.lambda - lambda_)*(xp_.lambda - lambda_) <= T(0.0))
        {
            return(true);
        }
        else
        {
            return(false);
        }

    }

    bool interpolate_solutions(const T& lambda_star)
    {
        T w = (lambda_star - lambda0)/(lambda1 - lambda0);
        T _w = T(1) - w;
        vec_ops->add_mul(_w, x0, w, x1);
        lambda1 = lambda_star;
        bool res = get_solution(lambda_star, x1);
        return(res);
    }


    bool get_solution(const T& lambda_fix, T_vec& x_)
    {
        bool converged;
        converged = newton->solve(nlin_op, x_, lambda_fix);
        return(converged);
    }


};

}


#endif // __BIFURCATION_DIAGRAM_CURVE_H__
