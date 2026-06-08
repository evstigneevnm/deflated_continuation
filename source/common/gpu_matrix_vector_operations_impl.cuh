#ifndef __gpu_matrix_vector_operations_impl_CUH__
#define __gpu_matrix_vector_operations_impl_CUH__


#include <cuda_runtime.h>
#include <utils/cuda_support.h>
#include <common/gpu_matrix_vector_operations.h>
#include <common/macros.h>
#include <stdexcept>



template<typename T>
__global__ void get_matrix_column_kernel(int Row, int Col, const T* matrix, T* vec, int col_number)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;  

    if((i<Row)&&(col_number<Col)&&(col_number>=0))
    {
        vec[i] = matrix[I2_R(i,col_number,Row)];
    }
}



template<typename T, typename T_vec, int BLOCKSIZE>
void gpu_matrix_vector_operations<T, T_vec, BLOCKSIZE>::get_matrix_column(vector_type vec, const matrix_type mat,  const size_t col_number)
{

    dim3 threads(BLOCKSIZE);
    int blocks_x=(sz_row+BLOCKSIZE)/BLOCKSIZE;
    dim3 blocks(blocks_x);
    
    get_matrix_column_kernel<scalar_type><<<blocks, threads>>>(sz_row, sz_col, mat, vec, col_number);

}


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
void gpu_matrix_vector_operations<T, T_vec, BLOCKSIZE>::set_matrix_column(matrix_type mat, const vector_type vec, const size_t col_number)
{

    dim3 threads(BLOCKSIZE);
    int blocks_x=(sz_row+BLOCKSIZE)/BLOCKSIZE;
    dim3 blocks(blocks_x);
    
    set_matrix_column_kernel<scalar_type><<<blocks, threads>>>(sz_row, sz_col, mat, vec, col_number);

}


template<typename T, typename T_vec, int BLOCKSIZE>
void gpu_matrix_vector_operations<T, T_vec, BLOCKSIZE>::set_matrix_value(matrix_type mat, const scalar_type val, const size_t row_number, const size_t col_number)
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

template<typename T>
__global__ void assign_kernel(int Row, int Col, const T* from_, T* to_)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;  

    if(i<Row*Col)
    {
        to_[i]=from_[i];
    }
}

template<typename T, typename T_vec, int BLOCKSIZE>
void gpu_matrix_vector_operations<T, T_vec, BLOCKSIZE>::assign(const matrix_type from_, matrix_type to_)
{

    dim3 threads(BLOCKSIZE);
    int blocks_x=(sz_row*sz_col+BLOCKSIZE)/BLOCKSIZE;
    dim3 blocks(blocks_x);
    
    assign_kernel<scalar_type><<<blocks, threads>>>(sz_row, sz_col, from_, to_);

}


template<typename T>
__global__ void make_zero_columns_kernel(size_t Rows, size_t Cols, const T* from_, size_t col_from, size_t col_to, T* to_)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i>=Rows) return;
    if(j>=Cols) return;

    to_[I2_R(i,j,Rows)] = from_[I2_R(i,j,Rows)];
    if((j >= col_from)&&(j < col_to))
    {
        to_[I2_R(i,j,Rows)] = T(0.0);
    }

}

template<typename T, typename T_vec, int BLOCKSIZE>
void gpu_matrix_vector_operations<T, T_vec, BLOCKSIZE>::make_zero_columns(const matrix_type from_, size_t col_from, size_t col_to, matrix_type to_)
{
    int BLOCKSIZE_y = 16;
    int BLOCKSIZE_x = BLOCKSIZE/BLOCKSIZE_y;

    dim3 dimBlock(BLOCKSIZE_x, BLOCKSIZE_y);
    dim3 dimGrid( (sz_row+BLOCKSIZE_x)/dimBlock.x,  (sz_col+BLOCKSIZE_y)/dimBlock.y);
    
    make_zero_columns_kernel<scalar_type><<<dimGrid, dimBlock>>>(sz_row, sz_col, from_, col_from, col_to, to_);

}


#endif