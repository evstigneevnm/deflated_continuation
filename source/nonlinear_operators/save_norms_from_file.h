#ifndef __NONLINEAR_OPERATORS__SAVE_NORMS_FROM_FILE__
#define __NONLINEAR_OPERATORS__SAVE_NORMS_FROM_FILE__

#include <common/file_operations.h>

namespace nonlinear_operators{


template<class VectorOperations, class VectorFileOperations, class NonlinearOperator>
class save_norms_from_file
{
public:
    using scalar_type = typename VectorOperations::scalar_type;
    using vector_type = typename VectorOperations::vector_type;

    using T = scalar_type;
    using T_vec = vector_type;

    save_norms_from_file(VectorOperations* vec_ops, VectorFileOperations* file_ops, NonlinearOperator* nonlin_op): 
    vec_ops_(vec_ops),
    file_ops_(file_ops),
    nonlin_op_(nonlin_op)
    {
        vec_ops_->init_vector(x); vec_ops_->start_use_vector(x);
    };

    ~save_norms_from_file()
    {
        vec_ops_->stop_use_vector(x); vec_ops_->free_vector(x);
    }
    

    void save_norms_all_files(const std::string& save_file_name, T param, const std::string& path, const std::string& regex_mask) const
    {
        auto solution_files = file_operations::match_file_names(path, regex_mask);
        std::ofstream f(save_file_name, std::ofstream::out);
        if (!f) throw std::runtime_error("nonlinear_operators:save_norms_from_file: error while opening file " + save_file_name);        
        for(auto &v: solution_files)
        {
            file_ops_->read_vector(v, (T_vec&)x);
            std::vector<T> bifurcaton_norms;
            nonlin_op_->norm_bifurcation_diagram(x, bifurcaton_norms);
            
            
            f << std::setprecision(16) << param << " ";
            for(auto &y: bifurcaton_norms)
            {
                f << std::setprecision(16) << y << " ";  //print all avaliable norms!
            }   
            f << v << std::endl;
        }
    }



private:
    VectorOperations* vec_ops_;
    VectorFileOperations* file_ops_;
    NonlinearOperator* nonlin_op_;
    T_vec x;





};

}

#endif