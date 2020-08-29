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

    
    curve_t curve;
    std::vector<curve_t> container_curves;

    
    int current_curve_number = 0;
    uint64_t id_file_name = 0;
    std::string project_dir;
    std::string curve_path;
    bool curve_opened = false;


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
        curve_path = project_dir.c_str() + std::to_string(current_curve_number);
        curve.clear();
        curve_opened = true;

    }

    void add(T lambda_, int unstable_dim_R_, int unstable_dim_C_, T_vec x_data = nullptr)
    {
        
        if(curve_opened)
        {
            bool is_data_avaliable = false;
            if(x_data != nullptr)
            {
                is_data_avaliable = true;
                id_file_name++;
                std::string f_name = curve_path.c_str()+std::string("/") + std::string("s") + std::to_string(id_file_name);
                vec_file_ops->write_vector(f_name, x_data);
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
        }
        else
        {
            throw std::runtime_error(std::string("stability_diagram: trying to add to a closed curve.") );            
        }
    }

    void close_curve()
    {
        id_file_name = 0;
        print_curve();
        curve_opened = false;
        container_curves.push_back(curve);
        
    }
    
    int current_curve()
    {
        return container_curves.size();
    }

    void pop_back_curve()
    {
        if(container_curves.size() > 0)
        {
            container_curves.pop_back();
            current_curve_number--;
        }
    }
    void print_curves_status()
    {
        
        std::cout << "container::stability_diagram current curve number = " << current_curve_number << std::endl;
        int cn_ = 0;
        for(auto &x: container_curves)
        {   
            std::cout << "container::stability_diagram curve (" << cn_ << "): has " << x.size() << " points." << std::endl;
            cn_++;
        }  

    }

    //makes a copy of a vector
    std::vector<stability_point_type> get_curve_points_vector(int curve_number_)
    {
        try
        {
            auto &curve = container_curves.at(curve_number_);
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
        ar & current_curve_number;  //a curve number should be serialized!!!
        ar & container_curves;
    }


    void print_curve()
    {
        std::string f_name = curve_path + std::string("/") + std::string("debug_curve_stability.dat");
        std::ofstream f(f_name.c_str(), std::ofstream::out);
        for(auto &x: curve)
        {
            f << x.lambda << " ";
            f << x.point_type << " ";
            f << x.unstable_dim_R << " ";
            f << x.unstable_dim_C << " ";
            f << x.id_file_name << std::endl;
        }
        f.close();
        log->info_f("container::stability_diagram(%i): printed final stability curve data.", current_curve_number); 
    }


};

}

#endif // __STABILITY_DIAGRAM_H__