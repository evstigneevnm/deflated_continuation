#ifndef __STABILITY_DIAGRAM_H__
#define __STABILITY_DIAGRAM_H__

/**
*
*   Class that implements stability diagram serrialization and output
*
*/

#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <system_error>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <utility>
//using boost for serialization
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>


namespace container
{

template<class T>
struct record_stability
{
    T lambda;
    bool is_data_avaliable = false;
    std::string point_type; //''stable'', ''unstable'', ''bifurcation''
    int unstable_dim_R;
    int unstable_dim_C;
    uint64_t id_file_name;

private:
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & lambda;
        ar & is_data_avaliable;
        ar & point_type;
        ar & unstable_dim_R;
        ar & unstable_dim_C;
        ar & id_file_name;                        
    }

};

template<class T>
struct stability_plot_record
{
    uint64_t source_point_index = 0;
    T lambda = T{};
    std::string point_type;
    int unstable_dim_R = 0;
    int unstable_dim_C = 0;
    int before_dim_R = 0;
    int before_dim_C = 0;
    int after_dim_R = 0;
    int after_dim_C = 0;
    std::string event_type = "none";
    uint64_t event_id = 0;
    std::vector<T> norms;
};

template<class T>
struct stability_topology_record
{
    uint64_t event_id = 0;
    uint64_t left_source_point = 0;
    uint64_t right_source_point = 0;
    T left_parameter = T{};
    T right_parameter = T{};
    T state_parameter = T{};
    T aligned_relative_distance = T{};
    int before_dim_R = 0;
    int before_dim_C = 0;
    int after_dim_R = 0;
    int after_dim_C = 0;
    std::string left_state_file;
    std::string right_state_file;
};



template<class VectorOperations, class VectorFileOperations, class Log>
class stability_diagram
{
private:
    friend class boost::serialization::access;
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;

public:
    typedef record_stability<T> stability_point_type;
private:
    typedef std::vector<stability_point_type> curve_t;
    typedef stability_plot_record<T> stability_plot_point_type;
    typedef std::vector<stability_plot_point_type> plot_curve_t;
    typedef stability_topology_record<T> stability_topology_point_type;
    typedef std::vector<stability_topology_point_type> topology_curve_t;

    
    curve_t curve;
    plot_curve_t plot_curve;
    topology_curve_t topology_curve;
    std::vector<curve_t> container_curves;

    
    int current_curve_number = 0;
    uint64_t id_file_name = 0;
    std::string project_dir;
    std::string curve_path;
    bool curve_opened = false;
    std::vector<std::filesystem::path> pending_files;


public:
   
    stability_diagram(VectorOperations* vec_ops_, VectorFileOperations* vec_file_ops_, Log* log_, const std::string project_dir_):
    log(log_),
    vec_ops(vec_ops_),
    vec_file_ops(vec_file_ops_),
    project_dir(project_dir_)
    {

    }

private:
    VectorOperations* vec_ops;
    VectorFileOperations* vec_file_ops;
    Log* log;

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
        if(directory_exists(curve_path))
        {
            return;
        }

        if(path_exists(curve_path))
        {
            throw std::runtime_error(
                std::string("stability_diagram: curve output path exists but is not a directory: ") +
                curve_path);
        }

        std::error_code ec;
        if(!std::filesystem::create_directory(curve_path, ec))
        {
            if(ec)
            {
                throw std::runtime_error(
                    std::string("stability_diagram: failed to create curve output directory '") +
                    curve_path + "': " + ec.message());
            }
            if(!directory_exists(curve_path))
            {
                throw std::runtime_error(
                    std::string("stability_diagram: failed to create curve output directory: ") +
                    curve_path);
            }
        }
        log->info_f("stability_diagram: created curve output directory: %s", curve_path.c_str());
    }

public:
    stability_diagram()
    {   
        //should be void for boost serrialization
    }
    ~stability_diagram()
    {

    }

    void open_curve(int cirve_number_)
    {
        current_curve_number = cirve_number_;
        if(container_curves.size() !=  current_curve_number)
        {
            throw std::runtime_error(std::string("stability_diagram: container size and curve number don't match.") );
        }
        curve_path = (std::filesystem::path(project_dir) / std::to_string(current_curve_number)).string();
        if(!directory_exists(project_dir))
        {
            if(path_exists(project_dir))
            {
                throw std::runtime_error(
                    std::string("stability_diagram: project path exists but is not a directory: ") +
                    project_dir);
            }
            throw std::runtime_error(
                std::string("stability_diagram: project directory doesn't exist: ") +
                project_dir);
        }
        ensure_curve_directory_exists();
        curve.clear();
        plot_curve.clear();
        topology_curve.clear();
        pending_files.clear();
        curve_opened = true;

    }

    void add(
        T lambda_,
        int unstable_dim_R_,
        int unstable_dim_C_)
    {
        add_record(
            lambda_,
            unstable_dim_R_,
            unstable_dim_C_,
            nullptr);
    }

    void add(
        T lambda_,
        int unstable_dim_R_,
        int unstable_dim_C_,
        const T_vec& x_data)
    {
        add_record(
            lambda_,
            unstable_dim_R_,
            unstable_dim_C_,
            &x_data);
    }

    void add_with_plot_data(
        T lambda_,
        int unstable_dim_R_,
        int unstable_dim_C_,
        uint64_t source_point_index_,
        const std::vector<T>& norms_,
        std::pair<int, int> before_dimension_,
        std::pair<int, int> after_dimension_)
    {
        const stability_point_type& point = add_record(
            lambda_,
            unstable_dim_R_,
            unstable_dim_C_,
            nullptr);
        add_plot_record(
            point,
            source_point_index_,
            norms_,
            before_dimension_,
            after_dimension_);
    }

    void add_topology_break_with_plot_data(
        T state_parameter,
        uint64_t left_source_point,
        uint64_t right_source_point,
        T left_parameter,
        T right_parameter,
        T aligned_relative_distance,
        const std::vector<T>& norms,
        std::pair<int, int> before_dimension,
        std::pair<int, int> after_dimension,
        const T_vec& left_state,
        const T_vec& right_state)
    {
        if(!curve_opened)
        {
            throw std::runtime_error(
                "stability_diagram: trying to add a topology break to "
                "a closed curve");
        }

        ++id_file_name;
        const std::string stem = "s" + std::to_string(id_file_name);
        const std::string left_file =
            (std::filesystem::path(curve_path)/stem).string();
        const std::string right_file =
            (std::filesystem::path(curve_path)/(stem + "_right")).string();
        vec_file_ops->write_vector(left_file, left_state);
        pending_files.emplace_back(left_file);
        vec_file_ops->write_vector(right_file, right_state);
        pending_files.emplace_back(right_file);

        stability_point_type point;
        point.lambda = state_parameter;
        point.is_data_avaliable = true;
        point.point_type = "topology_break";
        point.unstable_dim_R = after_dimension.first;
        point.unstable_dim_C = after_dimension.second;
        point.id_file_name = id_file_name;
        curve.push_back(point);

        stability_plot_point_type plot_point;
        plot_point.source_point_index = right_source_point;
        plot_point.lambda = state_parameter;
        plot_point.point_type = "topology_break";
        plot_point.unstable_dim_R = after_dimension.first;
        plot_point.unstable_dim_C = after_dimension.second;
        plot_point.before_dim_R = before_dimension.first;
        plot_point.before_dim_C = before_dimension.second;
        plot_point.after_dim_R = after_dimension.first;
        plot_point.after_dim_C = after_dimension.second;
        plot_point.event_type = "topology";
        plot_point.event_id = id_file_name;
        plot_point.norms = norms;
        plot_curve.push_back(std::move(plot_point));

        stability_topology_point_type topology;
        topology.event_id = id_file_name;
        topology.left_source_point = left_source_point;
        topology.right_source_point = right_source_point;
        topology.left_parameter = left_parameter;
        topology.right_parameter = right_parameter;
        topology.state_parameter = state_parameter;
        topology.aligned_relative_distance = aligned_relative_distance;
        topology.before_dim_R = before_dimension.first;
        topology.before_dim_C = before_dimension.second;
        topology.after_dim_R = after_dimension.first;
        topology.after_dim_C = after_dimension.second;
        topology.left_state_file = stem;
        topology.right_state_file = stem + "_right";
        topology_curve.push_back(std::move(topology));
    }

    void add_with_plot_data(
        T lambda_,
        int unstable_dim_R_,
        int unstable_dim_C_,
        uint64_t source_point_index_,
        const std::vector<T>& norms_,
        std::pair<int, int> before_dimension_,
        std::pair<int, int> after_dimension_,
        const T_vec& x_data)
    {
        const stability_point_type& point = add_record(
            lambda_,
            unstable_dim_R_,
            unstable_dim_C_,
            &x_data);
        add_plot_record(
            point,
            source_point_index_,
            norms_,
            before_dimension_,
            after_dimension_);
    }

    bool update_regular_point_dimension(
        uint64_t source_point_index,
        std::pair<int, int> dimension)
    {
        if(curve.size() != plot_curve.size())
        {
            throw std::runtime_error(
                "stability_diagram: stability and plot curve sizes "
                "do not match");
        }
        for(std::size_t offset = 0;
            offset < plot_curve.size();
            ++offset)
        {
            const std::size_t index =
                plot_curve.size() - offset - 1;
            auto& plot_point = plot_curve[index];
            auto& point = curve[index];
            if(
                plot_point.source_point_index != source_point_index ||
                plot_point.event_type != "none" ||
                point.is_data_avaliable)
                continue;

            point.unstable_dim_R = dimension.first;
            point.unstable_dim_C = dimension.second;
            point.point_type =
                dimension.first + dimension.second == 0
                ? "stable"
                : "unstable";
            plot_point.point_type = point.point_type;
            plot_point.unstable_dim_R = dimension.first;
            plot_point.unstable_dim_C = dimension.second;
            plot_point.before_dim_R = dimension.first;
            plot_point.before_dim_C = dimension.second;
            plot_point.after_dim_R = dimension.first;
            plot_point.after_dim_C = dimension.second;
            return true;
        }
        return false;
    }

private:
    const stability_point_type& add_record(
        T lambda_,
        int unstable_dim_R_,
        int unstable_dim_C_,
        const T_vec* x_data)
    {
        if(curve_opened)
        {
            bool is_data_avaliable = false;
            if(x_data != nullptr)
            {
                is_data_avaliable = true;
                id_file_name++;
                std::string f_name = curve_path.c_str()+std::string("/") + std::string("s") + std::to_string(id_file_name);
                vec_file_ops->write_vector(f_name, *x_data);
                pending_files.emplace_back(f_name);
                log->info_f("saved file: %s", f_name.c_str());

            }
            
            stability_point_type rec;

            rec.id_file_name = 0;

            if(x_data != nullptr)
            {
                rec.point_type = "bifurcation";
                rec.id_file_name = id_file_name;
            }
            else if(unstable_dim_R_+unstable_dim_C_==0)
            {
                rec.point_type = "stable";
            }
            else
            {
                rec.point_type = "unstable";
            }


            rec.is_data_avaliable = is_data_avaliable;
            rec.lambda = lambda_;
            rec.unstable_dim_R = unstable_dim_R_;
            rec.unstable_dim_C = unstable_dim_C_;

            curve.push_back(rec);
            return curve.back();
        }
        else
        {
            throw std::runtime_error(std::string("stability_diagram: trying to add to a closed curve.") );            
        }
    }

    static std::string classify_event(
        const stability_point_type& point,
        std::pair<int, int> before_dimension,
        std::pair<int, int> after_dimension)
    {
        if(point.point_type != "bifurcation")
            return "none";

        const int real_change =
            after_dimension.first - before_dimension.first;
        const int complex_change =
            after_dimension.second - before_dimension.second;
        if(real_change != 0 && complex_change == 0)
            return "steady";
        if(real_change == 0 && complex_change != 0)
            return "hopf";
        return "multiple";
    }

    void add_plot_record(
        const stability_point_type& point,
        uint64_t source_point_index,
        const std::vector<T>& norms,
        std::pair<int, int> before_dimension,
        std::pair<int, int> after_dimension)
    {
        stability_plot_point_type record;
        record.source_point_index = source_point_index;
        record.lambda = point.lambda;
        record.point_type = point.point_type;
        record.unstable_dim_R = point.unstable_dim_R;
        record.unstable_dim_C = point.unstable_dim_C;
        record.before_dim_R = before_dimension.first;
        record.before_dim_C = before_dimension.second;
        record.after_dim_R = after_dimension.first;
        record.after_dim_C = after_dimension.second;
        record.event_type = classify_event(
            point,
            before_dimension,
            after_dimension);
        record.event_id = point.id_file_name;
        record.norms = norms;
        plot_curve.push_back(std::move(record));
    }

public:
    void close_curve()
    {
        if(!curve_opened)
        {
            throw std::runtime_error(
                "stability_diagram: trying to close a closed curve");
        }
        id_file_name = 0;
        print_curve();
        curve_opened = false;
        container_curves.push_back(curve);
        curve.clear();
        plot_curve.clear();
        topology_curve.clear();
        pending_files.clear();
    }

    void abandon_curve()
    {
        for(const auto& file : pending_files)
        {
            std::error_code error;
            std::filesystem::remove(file, error);
            if(error)
            {
                log->warning_f(
                    "stability_diagram: failed to remove aborted "
                    "curve file %s: %s",
                    file.string().c_str(),
                    error.message().c_str());
            }
        }
        id_file_name = 0;
        curve.clear();
        plot_curve.clear();
        topology_curve.clear();
        pending_files.clear();
        curve_opened = false;
    }
    
    std::size_t current_curve() const
    {
        return container_curves.size();
    }

    std::size_t curve_count() const
    {
        return container_curves.size();
    }

    bool is_curve_open() const
    {
        return curve_opened;
    }

    void pop_back_curve()
    {
        if(!container_curves.empty())
        {
            container_curves.pop_back();
            current_curve_number =
                static_cast<int>(container_curves.size());
        }
    }
    void print_curves_status()
    {
        
        std::cout
            << "container::stability_diagram completed curves = "
            << container_curves.size() << std::endl;
        int cn_ = 0;
        for(auto &x: container_curves)
        {   
            std::cout << "container::stability_diagram curve (" << cn_ << "): has " << x.size() << " points." << std::endl;
            cn_++;
        }  

    }

    //makes a copy of a vector
    std::vector<stability_point_type> get_curve_points_vector(
        int curve_number_) const
    {
        try
        {
            const auto& curve = container_curves.at(curve_number_);
            return( curve );
        }
        catch(const std::exception& e)
        {
            log->warning_f("container::stability_diagram::get_curve_points_vector: %s", e.what());
            std::vector<stability_point_type> zero;
            return(zero);
        }
    }

    void get_solution_from_record(const int curve_number_, const stability_point_type& stability_point_, T_vec& x_)
    {
        if(stability_point_.is_data_avaliable)
        {
            std::string f_name = project_dir + std::to_string(curve_number_) + "/s" + std::to_string(stability_point_.id_file_name);
            vec_file_ops->read_vector(f_name, x_);
        }
        else
        {
            throw std::runtime_error(std::string("stability_diagram: get_solution_from_record: given point diesn't contain data.") );
        }

    }

private:

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & container_curves;
    }


    void print_curve()
    {
        print_legacy_curve();
        print_plot_curve();
        print_topology_curve();

        log->info_f(
            "container::stability_diagram(%i): printed final stability "
            "curve data.",
            current_curve_number);
    }

    void print_legacy_curve()
    {
        const std::filesystem::path file_name =
            std::filesystem::path(curve_path) /
            "debug_curve_stability.dat";
        const std::filesystem::path temporary(
            file_name.string() + ".tmp");
        std::error_code error;
        std::filesystem::remove(temporary, error);

        std::ofstream output(temporary, std::ofstream::out);
        if(!output)
        {
            throw std::runtime_error(
                "stability_diagram: failed to open temporary curve file: " +
                temporary.string());
        }

        for(const auto& point : curve)
        {
            output << point.lambda << " ";
            output << point.point_type << " ";
            output << point.unstable_dim_R << " ";
            output << point.unstable_dim_C << " ";
            output << point.id_file_name << '\n';
        }
        output.flush();
        output.close();
        if(!output)
        {
            std::filesystem::remove(temporary, error);
            throw std::runtime_error(
                "stability_diagram: curve output did not complete: " +
                temporary.string());
        }

        error.clear();
        std::filesystem::rename(temporary, file_name, error);
        if(error)
        {
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
            throw std::runtime_error(
                "stability_diagram: failed to replace curve file '" +
                file_name.string() + "': " + error.message());
        }
    }

    void print_plot_curve()
    {
        const std::filesystem::path file_name =
            std::filesystem::path(curve_path) /
            "debug_curve_stability_plot.dat";
        if(plot_curve.empty())
        {
            std::error_code error;
            std::filesystem::remove(file_name, error);
            if(error)
            {
                throw std::runtime_error(
                    "stability_diagram: failed to remove stale plot "
                    "sidecar '" + file_name.string() + "': " +
                    error.message());
            }
            return;
        }

        const std::filesystem::path temporary(
            file_name.string() + ".tmp");
        std::error_code error;
        std::filesystem::remove(temporary, error);

        std::ofstream output(temporary, std::ofstream::out);
        if(!output)
        {
            throw std::runtime_error(
                "stability_diagram: failed to open temporary plot "
                "sidecar: " + temporary.string());
        }

        output
            << "# stability_plot_v1\n"
            << "# source_index lambda point_type unstable_real "
               "unstable_complex_pairs before_real "
               "before_complex_pairs after_real after_complex_pairs "
               "event_type event_id norm_count norms...\n";
        output << std::setprecision(
            std::numeric_limits<T>::max_digits10);
        for(const auto& point : plot_curve)
        {
            output
                << point.source_point_index << ' '
                << point.lambda << ' '
                << point.point_type << ' '
                << point.unstable_dim_R << ' '
                << point.unstable_dim_C << ' '
                << point.before_dim_R << ' '
                << point.before_dim_C << ' '
                << point.after_dim_R << ' '
                << point.after_dim_C << ' '
                << point.event_type << ' '
                << point.event_id << ' '
                << point.norms.size();
            for(const T norm : point.norms)
                output << ' ' << norm;
            output << '\n';
        }
        output.flush();
        output.close();
        if(!output)
        {
            std::filesystem::remove(temporary, error);
            throw std::runtime_error(
                "stability_diagram: plot sidecar output did not "
                "complete: " + temporary.string());
        }

        error.clear();
        std::filesystem::rename(temporary, file_name, error);
        if(error)
        {
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
            throw std::runtime_error(
                "stability_diagram: failed to replace plot sidecar '" +
                file_name.string() + "': " + error.message());
        }
    }

    void print_topology_curve()
    {
        const std::filesystem::path file_name =
            std::filesystem::path(curve_path)/
            "debug_curve_stability_topology.dat";
        if(topology_curve.empty())
        {
            std::error_code error;
            std::filesystem::remove(file_name, error);
            if(error)
            {
                throw std::runtime_error(
                    "stability_diagram: failed to remove stale topology "
                    "sidecar '" + file_name.string() + "': " +
                    error.message());
            }
            return;
        }

        const std::filesystem::path temporary(
            file_name.string() + ".tmp");
        std::error_code error;
        std::filesystem::remove(temporary, error);
        std::ofstream output(temporary, std::ofstream::out);
        if(!output)
        {
            throw std::runtime_error(
                "stability_diagram: failed to open temporary topology "
                "sidecar: " + temporary.string());
        }
        output
            << "# stability_topology_v1\n"
            << "# event_id left_source right_source left_lambda "
               "right_lambda state_lambda aligned_relative_distance "
               "before_real before_complex_pairs after_real "
               "after_complex_pairs left_state_file right_state_file\n";
        output << std::setprecision(
            std::numeric_limits<T>::max_digits10);
        for(const auto& point : topology_curve)
        {
            output
                << point.event_id << ' '
                << point.left_source_point << ' '
                << point.right_source_point << ' '
                << point.left_parameter << ' '
                << point.right_parameter << ' '
                << point.state_parameter << ' '
                << point.aligned_relative_distance << ' '
                << point.before_dim_R << ' '
                << point.before_dim_C << ' '
                << point.after_dim_R << ' '
                << point.after_dim_C << ' '
                << point.left_state_file << ' '
                << point.right_state_file << '\n';
        }
        output.flush();
        output.close();
        if(!output)
        {
            std::filesystem::remove(temporary, error);
            throw std::runtime_error(
                "stability_diagram: topology sidecar output did not "
                "complete: " + temporary.string());
        }

        error.clear();
        std::filesystem::rename(temporary, file_name, error);
        if(error)
        {
            std::error_code ignored;
            std::filesystem::remove(temporary, ignored);
            throw std::runtime_error(
                "stability_diagram: failed to replace topology sidecar '" +
                file_name.string() + "': " + error.message());
        }
    }


};

}

#endif // __STABILITY_DIAGRAM_H__
