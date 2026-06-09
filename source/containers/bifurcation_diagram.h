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
#include <filesystem>
#include <fstream>
#include <system_error>
#include <type_traits>
#include <utility>
//using boost for serialization
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>

#include <containers/intersection_status.h>

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

    VectorOperations* vec_ops;
    VectorFileOperations* file_ops;
    Log* log;
    NonlinearOperator* nonlin_op;
    Newton* newton;
    cont_help_t* cont_help;
    unsigned int skip_output;
    std::string directory;
    bool legend_written = false;

public:
    typedef typename Curve::values_t curve_point_type;

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

    int current_curve()
    {
        return curve_container.size();
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

    std::vector<curve_point_type> get_curve_points_vector(int curve_number_)
    {
        try
        {
            auto &curve = curve_container.at(curve_number_);
            return( curve.return_curve_vector() );
        }
        catch(const std::exception& e)
        {
            log->warning_f("container::bifurcation_diagram::get_curve_points_vector: %s", e.what());
            std::vector<curve_point_type> zero;
            return(zero);
        }

    }


    std::pair<bool, bool> get_solutoin_from_curve(int& curve_number_, int& container_index_, T& lambda_p, T_vec& x_p)
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
            bool is_there_a_solution = curve.get_avalible_solution(container_index_, lambda_p, x_p);
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
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & curve_container;
        ar & skip_output;
        ar & curve_number;  //a curve number should be serialized!!!
        //ar & directory;      // should a directory be serialized?               
    }

    
};





}

#endif // __BIFURCATION_DIAGRAM_H__
