#pragma once //this will do for a test

#include <vector>
#include <string>

// include headers that implement a archive in simple text format
// #include <boost/archive/text_oarchive.hpp>
// #include <boost/archive/text_iarchive.hpp>
// for std::containers
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
// #include <boost/serialization/map.hpp>


template<class T>
struct complex_values
{
    friend class boost::serialization::access;
    
    T lambda;
    bool is_data_avaliable = false;
    std::vector<T> vector_norms;
    uint64_t id_file_name;

private:
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & lambda;
        ar & is_data_avaliable;
        ar & vector_norms;
        ar & id_file_name;                        
    }
    

};


template<class VecOps, class VecFiles>
class curve_container
{
    friend class boost::serialization::access;
    typedef typename VecOps::scalar_type T;
    typedef typename VecOps::vector_type T_vec;
public:

    void set_refs(VecOps* vec_ops_, VecFiles* vec_files_)
    {
        vec_ops = vec_ops_;
        vec_files = vec_files_;
    }

    curve_container(VecOps* vec_ops_, VecFiles* vec_files_, const std::string& directory_, int id_):
    vec_ops(vec_ops_),
    vec_files(vec_files_),
    directory(directory_),
    curve_id(id_)
    {
        open = true;
    }
    
    curve_container() //default constructor for boost
    {   
        std::cout << "curve_container default constructor" << std::endl;
    }

    ~curve_container()
    {
        std::cout << "curve_container distrctor" << std::endl;
    }

    // curve_container(const curve_container& that)
    // {
    //     *this = that;
    // }
    // curve_container& operator = (const curve_container& that)
    // {
    //     if(this == &that)
    //     {
    //         return *this;
    //     }
    //     else
    //     {
    //         container = that.container;
    ///                ...
    //         return *this;
    //     }
    // }
    curve_container(const curve_container&) = delete;
    curve_container& operator = (const curve_container& ) = delete;

    //allow move:
    curve_container(curve_container&& that)
    {
        //std::cout << "move '()': that = " << &that << " to this = " << this << std::endl;
        *this = std::move(that);
    }
    curve_container& operator =(curve_container&& that)
    {
        if(this == &that)
        {
            return *this;
        }
        else
        {
            std::cout << "move '=': that = " << &that << " to this = " << this << std::endl;
            container = std::move(that.container);
            directory = std::move(that.directory);
            vec_ops = that.vec_ops;
            vec_files = that.vec_files;
            global_id = that.global_id;
            curve_id = that.curve_id;
            open = that.open;
            return *this;
        }
    }


    void add(const T& lambda_, const T_vec& x_)
    {
        
        if(open)
        {
            T t1 = vec_ops->norm(x_);
            T t2 = vec_ops->norm_l2(x_);
            T t3 = vec_ops->norm_rank1(x_, lambda_);
            store(x_);

            values_t form_values;
            form_values.vector_norms.reserve(3);
            form_values.vector_norms.push_back(t1);
            form_values.vector_norms.push_back(t2);
            form_values.vector_norms.push_back(t3);

            form_values.lambda = lambda_;
            form_values.is_data_avaliable = true;
            form_values.id_file_name = global_id;
            container.push_back(form_values);
        }
        else
        {
            std::cout << "trying to add to a closed curve number " << curve_id << std::endl;
        }

    }

    void close()
    {
        container.shrink_to_fit();
        open = false;
    }

    void print()
    {
        std::cout << "curve = " << curve_id << " open = " << open << " vec_ops ref = " << vec_ops << std::endl;
        for(auto &x: container)
        {
            std::cout << "lambda = " << x.lambda << " id = " << x.id_file_name << " is data = " << x.is_data_avaliable << "  norms:( ";
            for(auto &y: x.vector_norms)
            {
                std::cout << y << " ";
            }
            std::cout << ")." << std::endl;
        }
        std::cout << "=X=X=X=X=X=X=X=X=X=X=" << std::endl;
    }

private:
    typedef complex_values<T> values_t;
    
    std::vector<values_t> container;
    std::string directory;
    VecOps* vec_ops;
    VecFiles* vec_files;
    uint64_t global_id = 0;
    int curve_id;
    bool open = false;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & container;
        //ar & directory;
        ar & global_id;
        ar & curve_id;   
        ar & open;
    }

    void store(const T_vec& x_)
    {
        global_id++;
        std::string f_name = directory.c_str()+std::string("/")+std::to_string(curve_id)+std::string("/")+std::to_string(global_id);
        std::cout << "container: file name = " << f_name << std::endl;
        vec_files->write_vector(f_name, x_);
    }

};



template<class VecOps, class VecFiles, class Curve>
class test_containers
{
    friend class boost::serialization::access;
    typedef typename VecOps::scalar_type T;
    typedef typename VecOps::vector_type T_vec;
public:
    
    test_containers(VecOps* vec_ops_, VecFiles* vec_files_, const std::string& directory_):
    vec_ops(vec_ops_),
    vec_files(vec_files_),
    directory(directory_),
    curve_number(-1)
    {

    }

    test_containers() //default constructor for boost
    {
        std::cout << "test_containers default constructor" << std::endl;
    }

    ~test_containers()
    {
        std::cout << "test_containers distrctor" << std::endl;
    }

    void get_current_ref(Curve*& curve_ref)
    {
        //std::cout << "reference to the curve inside = " << &curve_container.back() << std::endl;
        curve_ref = &curve_container.back();
    }

    void init_new_curve()
    {
        curve_number++;
        curve_container.emplace_back(vec_ops, vec_files, directory, curve_number) ;
    }

    void close_curve()
    {
        curve_container.back().close();
    }

    void add(const T& lambda_, const T_vec& x_)
    {
        curve_container.back().add(lambda_, x_);
    }

    void print_all()
    {
        for(auto &x: curve_container)
        {
            x.set_refs(vec_ops, vec_files);
            x.print();
        }
        std::cout << "reference address to the vec_ops = " << vec_ops << std::endl;
    }


private:
    std::vector<Curve> curve_container;
    std::string directory;
    VecOps* vec_ops;
    VecFiles* vec_files;
    int curve_number = -1;

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & curve_container;
        //ar & directory;
        ar & curve_number;
    }

    
};