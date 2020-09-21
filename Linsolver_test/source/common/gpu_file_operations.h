#ifndef __GPU_FILE_OPERATIONS_H__
#define __GPU_FILE_OPERATIONS_H__

#include <common/file_operations.h>


template<class VectorOperations>
class gpu_file_operations
{
public:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;

    gpu_file_operations(VectorOperations* vec_op_):
    vec_op(vec_op_)
    {
        sz = vec_op->get_vector_size();
    }

    ~gpu_file_operations()
    {

    }

    void write_vector(const std::string &f_name, const T_vec& vec_gpu, unsigned int prec=16) const
    {
        file_operations::write_vector<T>(f_name, sz, vec_op->view(vec_gpu), prec);
    }

    void read_vector(const std::string &f_name, T_vec vec_gpu) const
    {
        
        file_operations::read_vector<T>(f_name, sz, vec_op->view(vec_gpu));
        vec_op->set(vec_gpu);
    }


private:
    VectorOperations* vec_op;
    size_t sz;

};




#endif // __GPU_FILE_OPERATIONS_H__