#ifndef __ARNOLDI_file_operations_H__
#define __ARNOLDI_file_operations_H__

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <common/macros.h>


namespace file_operations
{

template <class T, class T_vec>
void write_2_vectors_by_side(const std::string &f_name, size_t N, const T_vec& vec1,  const T_vec& vec2, unsigned int prec=16, char sep = ' ')
{
    std::ofstream f(f_name.c_str(), std::ofstream::out);
    if (!f) throw std::runtime_error("print_vector: error while opening file " + f_name);

    for (size_t i = 0; i < N-1; ++i)
    {
        if (!(f << std::scientific << std::setprecision(prec) << vec1[i] << sep << vec2[i] <<  std::endl))
            throw std::runtime_error("print_vector: error while writing to file " + f_name);
    }
    if (!(f << std::scientific << std::setprecision(prec) << vec1[N-1] << sep << vec2[N-1] ))
        throw std::runtime_error("print_vector: error while writing to file " + f_name);
    
    f.close();
}


template <class T>
void write_vector(const std::string &f_name, size_t N, T *vec, unsigned int prec=19)
{
        std::ofstream f(f_name.c_str(), std::ofstream::out);
        if (!f) throw std::runtime_error("print_vector: error while opening file " + f_name);

        for (size_t i = 0; i < N-1; ++i)
        {
            if (!(f << std::scientific << std::setprecision(prec) << vec[i] <<  std::endl))
                throw std::runtime_error("print_vector: error while writing to file " + f_name);
        }
        if (!(f << std::scientific << std::setprecision(prec) << vec[N-1]))
            throw std::runtime_error("print_vector: error while writing to file " + f_name);
        
        f.close();
}
template <class T, class Vector>
void write_vector(const std::string &f_name, size_t N, Vector& vec, unsigned int prec=19)
{
        std::ofstream f(f_name.c_str(), std::ofstream::out);
        if (!f) throw std::runtime_error("print_vector: error while opening file " + f_name);

        for (size_t i = 0; i < N-1; ++i)
        {
            if (!(f << std::scientific << std::setprecision(prec) << vec[i] <<  std::endl))
                throw std::runtime_error("print_vector: error while writing to file " + f_name);
        }
        if (!(f << std::scientific << std::setprecision(prec) << vec[N-1]))
            throw std::runtime_error("print_vector: error while writing to file " + f_name);
        
        f.close();
}

template <class T_mat>
void write_matrix(const std::string &f_name, size_t Row, size_t Col, T_mat& matrix, unsigned int prec=19)
{
    size_t N=Col;
    std::ofstream f(f_name.c_str(), std::ofstream::out);
    if (!f) throw std::runtime_error("print_matrix: error while opening file " + f_name);
    for (size_t i = 0; i<Row; i++)
    {
        for(size_t j=0;j<Col;j++)
        {
            if(j<Col-1)
                f << std::scientific << std::setprecision(prec) << matrix[I2_R(i,j,Row)] << " ";
            else
                f << std::scientific << std::setprecision(prec) << matrix[I2_R(i,j,Row)];

        }
        f << std::endl;
    } 
    
    f.close();
}

inline std::pair<size_t, size_t> read_matrix_size(const std::string &f_name)
{

    std::ifstream f(f_name.c_str(), std::ifstream::in);
    if (!f) throw std::runtime_error("read_matrix_size: error while opening file " + f_name);
    std::string line;
    size_t matrix_size_rows = 0;
    size_t matrix_size_cols = 0;
    bool check_cols = true;

    while ( std::getline(f, line) )
    {
        if(check_cols)
        {
            for(auto &s: line)
            {
                if(s == ' ')
                {
                    ++matrix_size_cols;
                }                
            }
            check_cols = false;
        }
        ++matrix_size_rows;
    }
    f.close();
    return {matrix_size_rows, matrix_size_cols};
}

template <class T, class T_mat>
void read_matrix(const std::string &f_name,  size_t Row, size_t Col, T_mat& matrix){
    std::ifstream f(f_name.c_str(), std::ifstream::in);
    if (!f) throw std::runtime_error("read_matrix: error while opening file " + f_name);
    for (size_t i = 0; i<Row; i++)
    {
        for(size_t j=0;j<Col;j++)
        {
            // double val=0;  
            // fscanf(stream, "%le",&val);                
            // matrix[I2(i,j,Row)]=(real)val;
            T val;
            f >> val;
            matrix[I2_R(i,j,Row)]= static_cast<T>(val);
        }
        
    } 

    f.close();
}

template <class T>
int read_vector(const std::string &f_name,  size_t N,  T *vec){

    std::ifstream f(f_name.c_str(), std::ifstream::in);
    if (!f) throw std::runtime_error("read_vector: error while opening file " + f_name);
    for (size_t i = 0; i<N; i++)
    {
        T val;   
        f >> val;             
        vec[i]= static_cast<T>(val);           
    } 
    f.close();
    return 0;
}

template <class T, class Vector>
int read_vector(const std::string &f_name,  size_t N,  Vector& vec){

    std::ifstream f(f_name.c_str(), std::ifstream::in);
    if (!f) throw std::runtime_error("read_vector: error while opening file " + f_name);
    for (size_t i = 0; i<N; i++)
    {
        T val;   
        f >> val;             
        vec[i]= static_cast<T>(val);           
    } 
    f.close();
    return 0;
}


inline size_t read_vector_size(const std::string &f_name){

    std::ifstream f(f_name.c_str(), std::ifstream::in);
    if (!f) throw std::runtime_error("read_vector: error while opening file " + f_name);
    std::string line;
    size_t vector_size=0;
    while (std::getline(f, line))
    {
        vector_size++;
    }
    f.close();
    return vector_size;
}

}

#endif