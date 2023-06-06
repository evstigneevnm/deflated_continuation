#ifndef __GPU_FILE_OPERATIONS_H__
#define __GPU_FILE_OPERATIONS_H__

#include <common/file_operations.h>
#include <vector>

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

    void write_2_vectors_by_side(const std::string &f_name, const T_vec& vec1_gpu,  const T_vec& vec2_gpu, unsigned int prec=16, char sep = ' ') const
    {
        std::vector<T> vec1(sz);
        std::vector<T> vec2(sz);
        vec_op->get(vec1_gpu, vec1.data() );
        vec_op->get(vec2_gpu, vec2.data() );
        file_operations::write_2_vectors_by_side< T, std::vector<T> >(f_name, sz, vec1, vec2, prec, sep);
    }

    // size_t read_matrix_size(const std::string &f_name)
    // {
    //     return file_operations::read_matrix_size(f_name);
    // }

private:
    VectorOperations* vec_op;
    size_t sz;

};




#endif // __GPU_FILE_OPERATIONS_H__