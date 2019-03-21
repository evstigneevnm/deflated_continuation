#ifndef __GPU_FILE_OPERATIONS_H__
#define __GPU_FILE_OPERATIONS_H__

#include <utils/cuda_support.h>
#include <file_operations.h>



namespace gpu_file_operations
{
    template <class T>
    void write_vector(const std::string &f_name, size_t N, T*& vec_gpu, unsigned int prec=16)
    {
        T* vec_cpu = host_allocate<T>(N);
        device_2_host_cpy(vec_cpu, vec_gpu, N);
        file_operations::write_vector<T>(f_name, N, vec_cpu, prec);
        host_deallocate<T>(vec_cpu);
    }
    
    template <class T>
    void write_matrix(const std::string &f_name, size_t Row, size_t Col, T*& matrix_gpu, unsigned int prec=16)
    {  
        T* vec_cpu = host_allocate<T>(Row*Col);
        device_2_host_cpy(vec_cpu, matrix_gpu, Row*Col);
        file_operations::write_matrix<T>(f_name, Row, Col, vec_cpu, prec);
        host_deallocate<T>(vec_cpu);
    }

}


#endif