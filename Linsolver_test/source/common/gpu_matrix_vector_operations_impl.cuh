#ifndef __gpu_matrix_vector_operations_impl_CUH__
#define __gpu_matrix_vector_operations_impl_CUH__


#include <cuda_runtime.h>
#include <utils/cuda_support.h>
#include <common/gpu_matrix_vector_operations.h>
#include <common/macros.h>
#include <stdexcept>

template<typename T>
__global__ void set_matrix_column_kernel(int Row, int Col, T* matrix, const T* vec, int col_number)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;  

    if((i<Row)&&(col_number<Col)&&(col_number>=0))
    {
        matrix[I2_R(i,col_number,Row)]=vec[i];
    }
}



template<typename T, typename T_vec, int BLOCKSIZE>
void gpu_matrix_vector_operations<T, T_vec, BLOCKSIZE>::set_matrix_column(matrix_type& mat, const vector_type& vec, const size_t col_number)
{

    dim3 threads(BLOCKSIZE);
    int blocks_x=(sz_row+BLOCKSIZE)/BLOCKSIZE;
    dim3 blocks(blocks_x);
    
    set_matrix_column_kernel<scalar_type><<<blocks, threads>>>(sz_row, sz_col, mat, vec, col_number);

}


template<typename T, typename T_vec, int BLOCKSIZE>
void gpu_matrix_vector_operations<T, T_vec, BLOCKSIZE>::set_matrix_value(matrix_type& mat, const scalar_type& val, const size_t row_number, const size_t col_number)
{
    if((sz_row>row_number)&&(sz_col>col_number))
    {
        //assume that the matrix array is a continous memory block?!?
        //otherwise this will fail!
        matrix_type mat_value_ref = mat+I2_R(row_number, col_number, sz_row);
        host_2_device_cpy<T>(mat_value_ref, (T*)&val, 1);
    }
    else
    {
        throw std::runtime_error("set_matrix_value: invalid col or row to set a value: ");
    }

}



#endif