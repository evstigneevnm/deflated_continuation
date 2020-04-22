#ifndef __BIFURCATION_DIAGRAM_CURVE_H__
#define __BIFURCATION_DIAGRAM_CURVE_H__

#include <string>
#include <stdexcept>
#include <fstream>
#include <iostream>


namespace container
{

template<class T>
struct complex_values
{
    T lambda;
    bool is_data_avaliable = false;
    std::vector<T> vector_norms;
    uint64_t id_file_name;
};

template<class VectorOperations, class VectorFileOperations, class Log, class NonlinearOperator, class Newton, class SolutionStorage, class HelperVectors>
class bifurcation_diagram_curve
{
public:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;

    bifurcation_diagram_curve(VectorOperations*& vec_ops_, VectorFileOperations*& vec_files_, Log*& log_, NonlinearOperator*& nlin_op_, Newton*& newton_, int curve_number_, HelperVectors*& helper_vectors_):
    vec_ops(vec_ops_),
    vec_files(vec_files_),
    log(log_),
    nlin_op(nlin_op_),
    newton(newton_)
    {
        std::cout << "constructor of this " << this << " with "; //std::endl;
//NOTICE:
//it constructs these helper vectors every time we add this to the container
        //vec_ops->init_vector(x0); vec_ops->start_use_vector(x0);
        //vec_ops->init_vector(x1); vec_ops->start_use_vector(x1);
//too much ram will be used!
//a better solution is to use external vectors for this.
//but may couse logical problems?
//assume now that a HelperVectors class contains T_vec x0 and T_vec x1 and  can be accessed via reference.

        helper_vectors_->get_refs(x0, x1);
        std::cout << "x0 = " << x0 << " x1 = " << x1 << std::endl;
        set_curve_number(curve_number_);
    }
    
    ~bifurcation_diagram_curve()
    {
        std::cout << "distructor of this " << this << " with &x0 = " << x0 << " and &x1 = " << x1 << std::endl;
        //vec_ops->stop_use_vector(x0); vec_ops->free_vector(x0);
        //vec_ops->stop_use_vector(x1); vec_ops->free_vector(x1);

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
        std::cout << "move constructor of that " << &that << " to this " << this << std::endl;
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
            std::cout << "move assign of that " << &that << " to this " << this << std::endl;
            std::swap(vec_ops, that.vec_ops);
            std::swap(vec_files, that.vec_files);
            std::swap(log, that.log);
            std::swap(nlin_op, that.nlin_op);
            std::swap(newton, that.newton);
            std::swap(data_directory, that.data_directory);
            std::swap(container, that.container);
            std::swap(global_id, that.global_id);
            std::swap(x0, that.x0);
            std::swap(x1, that.x1);
            std::swap(lambda0, that.lambda0);
            std::swap(lambda1, that.lambda1);
            return *this;
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


    inline bool fs_object_exsts(const std::string& name) 
    {
        std::ifstream f(name.c_str());
        return f.good();
    }

public:



    void set_directory(const std::string& data_directory_)
    {
        data_directory = data_directory_;
        if(!fs_object_exsts(data_directory))
            throw std::runtime_error(std::string("container::bifurcation_diagram_curve: provided directory doesn't exist") );
    }

    void set_curve_number(int curve_number_)
    {
        curve_number = curve_number_;
        std::string full_path = data_directory+std::string("/")+std::to_string(curve_number);

        if(!fs_object_exsts(full_path) )
        {
//TODO For now. Later this is to be replaced by create directory
            throw std::runtime_error(std::string("container::bifurcation_diagram_curve: provided directory and curve_number doesn't exist") );
        }

    }

    void add(const T& lambda_, const T_vec& x_)
    {
        std::vector<T> bif_diag_norms;
        nlin_op->norm_bifurcation_diagram(x_, bif_diag_norms);
        store_t store_result = store(lambda_, x_);

        values_t form_values;
        form_values.lambda = lambda_;
        form_values.is_data_avaliable = store_result.first;
        form_values.id_file_name = store_result.second;
        form_values.vector_norms = bif_diag_norms;

        container.push_back(form_values);
    }

    bool find_intersection(const T& lambda_star, SolutionStorage& solution_vector)
    {
        int N = container.size();
        for(int j=0;j<N-1;j++)
        {
            int ind = j;
            int indp = j+1;
            auto &p_j = container[ind];
            auto &p_jp = container[indp];
            if(intersection(p_j, p_jp, lambda_star))
            {
                bool stat_l = get_lower(ind);
                bool stat_u = get_upper(indp);
                if(stat_l&&stat_u)
                {
                    if(interpolate_solutions(lambda_star))
                    {
                        solution_vector.push_back(x1);
                    }
                    else
                    {
                        throw std::runtime_error(std::string("container::bifurcation_diagram_curve: newton failed in solution section") );
                    }
                }
                else
                {
                    throw std::runtime_error(std::string("container::bifurcation_diagram_curve: failed to find valid solution") );
                }
            }
        }

    }


// Debug print out for gnuplot
    void print_curve()
    {
        std::string f_name = data_directory+std::string("/")+std::to_string(curve_number) + std::string("/") + std::string("debug_curve.dat");
        std::ofstream f(f_name.c_str(), std::ofstream::out);
        for(auto &x: container)
        {
            f << x.lambda << " ";
            for(auto &y: x.vector_norms)
            {
                f << y << " ";  //print all avaliable norms!
            }
            f << x.id_file_name << std::endl;
        }
        f.close();
    }

//TODO: function that adds to a specific file at every step for monitoring.
    

//TODO: add delete function that removes all information from the container for given interval of lambdas
    void delete_ponts(const int& lambda_min, const int& lambda_max)
    {

    }

private:
    typedef std::pair<bool, uint64_t> store_t;
    typedef complex_values<T> values_t;
    typedef std::vector<values_t> b_d_container_t;
    b_d_container_t container;
    uint64_t global_id = 0; 

    T_vec x0 = nullptr;
    T_vec x1 = nullptr;
    T lambda0, lambda1;

    store_t store(const T& lambda_, const T_vec& x_)
    {
        //TODO: add condition for storing data on the drive
        store_t res;
        res.first = true;
        
        if(res.first)
        {
            global_id++;
            
            std::string f_name = data_directory+std::string("/")+std::to_string(curve_number)+std::string("/")+std::to_string(global_id);
            vec_files->write_vector(f_name, x_);
            res.second = global_id;
        }
        else
        {
            res.second = 0;
        }
        return(res);
    }

    bool get_lower(int index)
    {

        int j = index;
        bool saved_data = false;
        while(saved_data)
        {
            values_t local_data = container[j];
            saved_data = local_data.is_data_avaliable;
            if(saved_data)
            {
                lambda0 = local_data.lambda;
                uint64_t local_id = local_data.id_file_name;
                vec_files->read_vector(std::string(local_id), x0);
                break;
            }
            j--;
            if(j<0)
                break;

        }
        return(saved_data);
        
    }
    bool get_upper(int index)
    {

        int j = index;
        bool saved_data = false;
        while(saved_data)
        {
            values_t local_data = container[j];
            saved_data = local_data.is_data_avaliable;
            if(saved_data)
            {
                lambda1 = local_data.lambda;
                uint64_t local_id = local_data.id_file_name;
                vec_files->read_vector(std::string(local_id), x1);
                break;
            }
            j++;
            if(j>=container.size())
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
        vec_ops->add_mul(w, x0, _w, x1);
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