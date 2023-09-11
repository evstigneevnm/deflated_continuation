#ifndef __CPU_FILE_OPERATIONS_H__
#define __CPU_FILE_OPERATIONS_H__

#include <common/file_operations.h>
#include <vector>

template<class VectorOperations>
class cpu_file_operations
{
public:
    typedef typename VectorOperations::scalar_type  T;
    typedef typename VectorOperations::vector_type  T_vec;

    cpu_file_operations(VectorOperations* vec_op_):
    vec_op(vec_op_)
    {
        sz = vec_op->get_vector_size();
        prec_ = vec_op->get_fp_prec();
    }

    ~cpu_file_operations()
    {}

    void write_vector(const std::string &f_name, T_vec& vec_cpu) const
    {
        file_operations::write_vector<T, T_vec>(f_name, sz, vec_cpu, prec_);
    }

    void read_vector(const std::string &f_name, const T_vec& vec_cpu) const
    {
        
        file_operations::read_vector<T, T_vec>(f_name, sz, vec_cpu);
    }

    void write_2_vectors_by_side(const std::string &f_name, const T_vec& vec1,  const T_vec& vec2, unsigned int prec=16, char sep = ' ') const
    {
        file_operations::write_2_vectors_by_side< T, T_vec >(f_name, sz, vec1, vec2, prec, sep);
    }

private:
    VectorOperations* vec_op;
    size_t sz;
    unsigned int prec_;

};




#endif