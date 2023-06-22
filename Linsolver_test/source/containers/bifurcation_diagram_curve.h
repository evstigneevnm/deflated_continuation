#ifndef __BIFURCATION_DIAGRAM_CURVE_H__
#define __BIFURCATION_DIAGRAM_CURVE_H__

#include <string>
#include <vector>
#include <stdexcept>
#include <fstream>
#include <iostream>


//using boost for serialization
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>


namespace container
{

template<class T>
struct complex_values
{
    T lambda;
    bool is_data_avaliable = false;
    std::vector<T> vector_norms;
    uint64_t id_file_name;

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
            curve_number = that.curve_number;
            x0 = that.x0;
            x1 = that.x1;
            lambda0 = that.lambda0;
            lambda1 = that.lambda1;
            skip_output = that.skip_output;
            debug_f_name  = std::move(that.debug_f_name);
            debug_file = std::move(that.debug_file); //std::move(that.debug_file). Move of std::ofstream supported only from C++5.X and above!
            curve_open = that.curve_open;
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
    std::ofstream debug_file;
    bool refs_set = false;


    inline bool fs_object_exsts(const std::string& name) 
    {
        std::ifstream f(name.c_str());
        return f.good();
    }

public:



    void set_directory(const std::string& data_directory_)
    {
        data_directory.assign(data_directory_);
        if(!fs_object_exsts(data_directory))
            throw std::runtime_error(std::string("container::bifurcation_diagram_curve: provided directory doesn't exist") );
    }


    void set_curve_number(int curve_number_)
    {
        curve_number = curve_number_;
        full_path.assign( data_directory.c_str()+std::to_string(curve_number) );
        debug_f_name.assign(full_path.c_str() + std::string("/") + std::string("debug_curve.dat"));
        log->info_f("container::bifurcation_diagram_curve: FULL PATH: %s", full_path.c_str());
        if(!fs_object_exsts(full_path) )
        {
//TODO For now. Later this is to be replaced by create directory
            throw std::runtime_error(std::string("container::bifurcation_diagram_curve: provided directory and curve_number doesn't exist") );
        }

    }

    void add(const T& lambda_, const T_vec& x_, bool force_store = false)
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
    void find_intersection(const T& lambda_star, SolutionStorage*& solution_vector)
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
                if((p_j.lambda == lambda_star)&&(p_j.is_data_avaliable))
                {
                    uint64_t local_id = p_j.id_file_name;
                    std::string f_name = full_path+std::string("/")+std::to_string(local_id);
                    vec_files->read_vector(f_name, x1); 
                    solution_vector->push_back(x1); 
                    log->info_f("container::bifurcation_diagram_curve(%i): added intersectoin at (%i) for the solution at lambda =  %lf", curve_number, ind, lambda_star);               
                }
                else if((p_jp.lambda == lambda_star)&&(p_jp.is_data_avaliable))
                {
                    uint64_t local_id = p_jp.id_file_name;
                    std::string f_name = full_path+std::string("/")+std::to_string(local_id);
                    vec_files->read_vector(f_name, x1); 
                    solution_vector->push_back(x1); 
                    log->info_f("container::bifurcation_diagram_curve(%i): added intersectoin at (%i) for the solution at lambda =  %lf", curve_number, indp, lambda_star);               

                }
                else
                {
                    bool stat_l = get_lower(ind);
                    bool stat_u = get_upper(indp);
                    if(stat_l&&stat_u)
                    {
                        if(interpolate_solutions(lambda_star))
                        {
                            solution_vector->push_back(x1);
                            log->info_f("container::bifurcation_diagram_curve(%i): added intersectoin at (%i,%i) for the solution at lambda =  %lf", curve_number, ind, indp, lambda_star);
                        }
                        else
                        {
                            //throw std::runtime_error(std::string("container::bifurcation_diagram_curve: newton failed in solution section") );
                            log->warning_f("container::bifurcation_diagram_curve(%i): !!!newton failed in solution section at (%i(%i), %i(%i)) for the solution at lambda =  %lf !!!", curve_number, ind, int(stat_l), indp, int(stat_u), lambda_star);
                        }
                    }
                    else
                    {
                        //std::string fail_find_files = std::string("container::bifurcation_diagram_curve(") + std::to_string(curve_number) + std::string("): failed to find a valid solution for the parameter = ") + std::to_string(lambda_star) + std::string(" with lower flag = ") + std::to_string(stat_l) + std::string(" and upper flag = ") + std::to_string(stat_u) + std::string(", indexing = (") + std::to_string(ind) + std::string(",") + std::to_string(indp) + std::string(").");
                        log->warning_f("container::bifurcation_diagram_curve(%i): !!!failed to add intersectoin at (%i(%i), %i(%i)) for the solution at lambda =  %lf !!!", curve_number, ind, int(stat_l), indp, int(stat_u), lambda_star);

                        //throw std::runtime_error( fail_find_files );
                    }
                }
            }
        }

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
                debug_file << x.lambda << " ";
                for(auto &y: x.vector_norms)
                {
                    debug_file << y << " ";  //print all avaliable norms!
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
            f << x.lambda << " ";
            for(auto &y: x.vector_norms)
            {
                f << y << " ";  //print all avaliable norms!
            }
            f << x.id_file_name << std::endl;
        }
        f.close();
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
        curve_open = false; 
        log->info_f("container::bifurcation_diagram_curve(%i) closed.", curve_number); 
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

public:
    typedef complex_values<T> values_t;
private:
    typedef std::pair<bool, uint64_t> store_t;
    typedef std::vector<values_t> b_d_container_t;
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

    T_vec x0 = nullptr;
    T_vec x1 = nullptr;
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

    bool get_lower(int index)
    {

        int j = index;
        bool saved_data = false;
        while(!saved_data)
        {
            values_t local_data = container[j];
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
    bool get_upper(int index)
    {

        int j = index;
        bool saved_data = false;
        while(!saved_data)
        {
            int container_size = container.size();
            //std::cout << "container_size = " << container_size << std::endl;
            
            values_t local_data = container[j];
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