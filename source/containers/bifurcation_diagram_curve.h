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
#include <algorithm>
#include <limits>
#include <utility>


//using boost for serialization
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>

#include <containers/branch_intersection.h>
#include <containers/curve_endpoint_reason.h>
#include <containers/intersection_status.h>
#include <containers/symmetry_event_journal.h>
#include <containers/bifurcation_diagram/curve_intersection_search.h>
#include <containers/bifurcation_diagram/curve_point.h>
#include <containers/bifurcation_diagram/curve_provenance.h>
#include <containers/bifurcation_diagram/curve_interpolator.h>
#include <containers/bifurcation_diagram/curve_metadata_io.h>
#include <containers/bifurcation_diagram/curve_vector_store.h>

namespace container
{

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
    typedef symmetry_event_record<T> symmetry_event_record_type;

private:
    using vector_store_type = curve_vector_store<VectorFileOperations, Log, T_vec>;
    using interpolator_type = curve_interpolator<
        VectorOperations,
        vector_store_type,
        NonlinearOperator,
        Newton,
        values_t>;
    using intersection_search_type = curve_intersection_search<
        VectorOperations,
        interpolator_type,
        values_t>;

public:
    

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
            vector_store.bind(vec_files, log);
            interpolator.bind(vec_ops, &vector_store, nlin_op, newton, &x0, &x1);
            intersection_search.bind(
                vec_ops,
                &interpolator,
                &container,
                &incomplete_segment_ids,
                &x0,
                &x1);
            refs_set = true;
            load_metadata_if_available();
            load_provenance_if_available();
            load_symmetry_events_if_available();
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
        vector_store.bind(vec_files, log);
        interpolator.bind(vec_ops, &vector_store, nlin_op, newton, &x0, &x1);
        intersection_search.bind(
            vec_ops,
            &interpolator,
            &container,
            &incomplete_segment_ids,
            &x0,
            &x1);
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
            skip_output = that.skip_output;
            debug_f_name  = std::move(that.debug_f_name);
            metadata_f_name = std::move(that.metadata_f_name);
            provenance_f_name = std::move(that.provenance_f_name);
            provenance = std::move(that.provenance);
            provenance_metadata_available = that.provenance_metadata_available;
            debug_file = std::move(that.debug_file); //std::move(that.debug_file). Move of std::ofstream supported only from C++5.X and above!
            curve_open = that.curve_open;
            refs_set = that.refs_set;
            current_segment_id = that.current_segment_id;
            current_semicurve_id = that.current_semicurve_id;
            segment_metadata_available = that.segment_metadata_available;
            incomplete_segment_ids = std::move(that.incomplete_segment_ids);
            symmetry_events = std::move(that.symmetry_events);
            vector_store.bind(vec_files, log);
            interpolator.bind(vec_ops, &vector_store, nlin_op, newton, &x0, &x1);
            intersection_search.bind(
                vec_ops,
                &interpolator,
                &container,
                &incomplete_segment_ids,
                &x0,
                &x1);
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
    std::string provenance_f_name;
    std::ofstream debug_file;
    bool refs_set = false;
    uint64_t current_segment_id = 0;
    uint64_t current_semicurve_id = 0;
    bool segment_metadata_available = false;
    std::vector<uint64_t> incomplete_segment_ids;
    symmetry_event_journal<T> symmetry_events;
    vector_store_type vector_store;
    interpolator_type interpolator;
    intersection_search_type intersection_search;
    curve_provenance provenance;
    bool provenance_metadata_available = false;

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
        if(!write_curve_metadata(metadata_f_name, container))
        {
            if(log != nullptr)
            {
                log->warning_f("container::bifurcation_diagram_curve(%i): failed to open metadata file %s", curve_number, metadata_f_name.c_str());
            }
        }
    }

    void load_metadata_if_available()
    {
        if(metadata_f_name.empty())
        {
            metadata_f_name.assign(full_path + std::string("/") + std::string("metadata_curve.dat"));
        }
        const auto result = load_curve_metadata(metadata_f_name, container);
        incomplete_segment_ids = result.incomplete_segment_ids;
        segment_metadata_available = result.loaded_any;
    }

    void write_provenance_file()
    {
        if(provenance_f_name.empty())
        {
            return;
        }
        if(!write_curve_provenance(provenance_f_name, provenance) && log != nullptr)
        {
            log->warning_f(
                "container::bifurcation_diagram_curve(%i): failed to write provenance file %s",
                curve_number,
                provenance_f_name.c_str());
        }
    }

    void load_provenance_if_available()
    {
        if(provenance_f_name.empty())
        {
            provenance_f_name.assign(
                (std::filesystem::path(full_path)/"curve_provenance.dat").string());
        }
        provenance_metadata_available =
            load_curve_provenance(provenance_f_name, provenance);
    }

    void load_symmetry_events_if_available()
    {
        symmetry_events.set_file_name(
            std::filesystem::path(full_path)/"symmetry_events.dat");
        if(!symmetry_events.load())
        {
            return;
        }
        for(auto& event: symmetry_events.all_mutable())
        {
            event.curve_number = curve_number;
            if(event.point_index < container.size())
            {
                const auto& point = container[static_cast<std::size_t>(event.point_index)];
                event.vector_available = point.is_data_avaliable;
                event.vector_file_id = point.id_file_name;
            }
        }
    }

    bool can_interpolate_between(const values_t& lower, const values_t& upper) const
    {
        if(!segment_metadata_available)
        {
            return true;
        }
        // An incomplete endpoint means that the segment could not be extended;
        // it does not invalidate the already accepted states inside the
        // segment. Only interpolation across segment boundaries is forbidden.
        return lower.segment_id == upper.segment_id;
    }

    bool is_incomplete_pair(const values_t& lower, const values_t& upper) const
    {
        return segment_metadata_available &&
               lower.segment_id == upper.segment_id &&
               is_incomplete_segment(lower.segment_id);
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
        provenance_f_name.assign(
            (std::filesystem::path(full_path)/"curve_provenance.dat").string());
        symmetry_events.set_file_name(
            std::filesystem::path(full_path)/"symmetry_events.dat");
        log->info_f("container::bifurcation_diagram_curve: FULL PATH: %s", full_path.c_str());
        ensure_curve_directory_exists();

    }

    void reset_output_directory(const std::string& data_directory_)
    {
        set_directory(data_directory_);
        set_curve_number(curve_number);
        load_metadata_if_available();
        load_provenance_if_available();
        load_symmetry_events_if_available();
    }

    void set_analytical_branch_provenance(
        const uint64_t branch_id,
        const std::string& branch_name)
    {
        provenance.origin = curve_origin::analytical;
        provenance.analytical_branch_id = branch_id;
        provenance.analytical_branch_name = branch_name;
        provenance_metadata_available = true;
        write_provenance_file();
    }

    const curve_provenance& get_curve_provenance() const
    {
        return provenance;
    }

    bool has_curve_provenance_metadata() const
    {
        return provenance_metadata_available;
    }

    int get_curve_number() const
    {
        return curve_number;
    }

    uint64_t get_current_segment_id() const
    {
        return current_segment_id;
    }

    std::size_t point_count() const
    {
        return container.size();
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

    void record_symmetry_intersection(
        const std::size_t previous_order,
        const std::size_t candidate_order,
        const T& previous_transverse_ratio,
        const T& candidate_transverse_ratio,
        const unsigned int refinements)
    {
        if(container.empty())
        {
            return;
        }
        const auto& point = container.back();
        symmetry_event_record<T> event;
        event.curve_number = curve_number;
        event.point_index = point.point_index;
        event.lambda = point.lambda;
        event.vector_available = point.is_data_avaliable;
        event.vector_file_id = point.id_file_name;
        event.segment_id = point.segment_id;
        event.semicurve_id = point.semicurve_id;
        event.previous_order = previous_order;
        event.candidate_order = candidate_order;
        event.previous_orbit_type =
            symmetry::translation::orbit_type::cyclic_1d(previous_order);
        event.candidate_orbit_type =
            symmetry::translation::orbit_type::cyclic_1d(candidate_order);
        event.previous_transverse_ratio = previous_transverse_ratio;
        event.candidate_transverse_ratio = candidate_transverse_ratio;
        event.refinements = refinements;
        symmetry_events.stage(event);
    }

    template<class Event>
    void record_symmetry_intersection(const Event& source)
    {
        if(container.empty())
        {
            return;
        }
        const auto& point = container.back();
        symmetry_event_record<T> event;
        event.curve_number = curve_number;
        event.point_index = point.point_index;
        event.lambda = point.lambda;
        event.vector_available = point.is_data_avaliable;
        event.vector_file_id = point.id_file_name;
        event.segment_id = point.segment_id;
        event.semicurve_id = point.semicurve_id;
        event.previous_order = source.previous_order;
        event.candidate_order = source.candidate_order;
        event.previous_orbit_type = source.previous_orbit_type;
        event.candidate_orbit_type = source.candidate_orbit_type;
        event.previous_transverse_ratio = source.previous_transverse_ratio;
        event.candidate_transverse_ratio = source.candidate_transverse_ratio;
        event.refinements = source.refinements;
        symmetry_events.stage(event);
    }

    bool commit_staged_symmetry_events()
    {
        if(!symmetry_events.has_staged_changes())
        {
            return true;
        }
        const bool committed = symmetry_events.commit();
        if(!committed)
        {
            log->warning_f(
                "container::bifurcation_diagram_curve(%i): failed to commit staged symmetry events",
                curve_number);
        }
        return committed;
    }

    const std::vector<symmetry_event_record<T>>& symmetry_event_records() const
    {
        return symmetry_events.all();
    }

    //return a solution pair (x,\lambda) from the container
    //for the stability analysis. 
    //Should be done as a querry operation.
    //container_index is returned to the upper level
    //returned 'true' means that the pair is found, 'false' - that there are no more pairs
    bool get_avalible_solution(int& container_index, T& lambda_p, T_vec& x_p)
    {
        values_t point;
        bool metadata_available = false;
        return get_avalible_solution(
            container_index,
            lambda_p,
            x_p,
            point,
            metadata_available);
    }

    bool get_avalible_solution(
        int& container_index,
        T& lambda_p,
        T_vec& x_p,
        values_t& point,
        bool& metadata_available)
    {
        int N = container.size();
        metadata_available = segment_metadata_available;
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
                    vector_store.read(full_path, local_id, x_p);
                    lambda_p = p_j.lambda;
                    point = p_j;
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
                    vector_store.read(full_path, local_id, x1);
                    solution_vector->push_back(x1); 
                    status.added++;
                    log->info_f("container::bifurcation_diagram_curve(%i): added intersectoin at (%i) for the solution at lambda =  %lf", curve_number, ind, lambda_star);               
                }
                else if((p_jp.lambda == lambda_star)&&(p_jp.is_data_avaliable))
                {
                    uint64_t local_id = p_jp.id_file_name;
                    vector_store.read(full_path, local_id, x1);
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
        return intersection_search.find_branch_intersection(
            step_lambda0,
            step_x0,
            step_lambda1,
            step_x1,
            step_norms0,
            step_norms1,
            policy,
            segment_metadata_available,
            full_path,
            curve_number,
            hit_x,
            result,
            std::forward<StateDistance>(state_distance));
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
        return intersection_search.find_self_intersection(
            step_lambda0,
            step_x0,
            step_lambda1,
            step_x1,
            step_norms0,
            step_norms1,
            policy,
            segment_metadata_available,
            full_path,
            curve_number,
            hit_x,
            result,
            std::forward<StateDistance>(state_distance));
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
        write_provenance_file();
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

    bool read_saved_solution_for_audit(
        const values_t& point,
        T_vec& output)
    {
        return read_saved_point(point, output);
    }

private:
    b_d_container_t container;
    uint64_t global_id = 0; 
    uint64_t global_index = 0;

    T_vec x0;
    T_vec x1;
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
        return vector_store.store(
            full_path,
            skip_output,
            global_index,
            global_id,
            x_,
            force_store_);
    }

    bool read_saved_point(const values_t& point, T_vec& x_out)
    {
        return interpolator.read_saved_point(full_path, point, x_out);
    }

    bool evaluate_segment_at_lambda(const int lower_index, const int upper_index, const T& lambda_star, T_vec& x_out)
    {
        return interpolator.evaluate_segment_at_lambda(
            container,
            lower_index,
            upper_index,
            lambda_star,
            segment_metadata_available,
            full_path,
            x_out);
    }

    bool get_lower(int index, uint64_t segment_id)
    {
        return interpolator.load_lower(
            container,
            index,
            segment_id,
            segment_metadata_available,
            full_path);
    }
    bool get_upper(int index, uint64_t segment_id)
    {
        return interpolator.load_upper(
            container,
            index,
            segment_id,
            segment_metadata_available,
            full_path);
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
        return interpolator.interpolate_prepared(lambda_star);
    }


};

}


#endif // __BIFURCATION_DIAGRAM_CURVE_H__
