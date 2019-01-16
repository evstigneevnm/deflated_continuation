#ifndef __ARNOLDI_file_operations_H__
#define __ARNOLDI_file_operations_H__

#include <cstdlib>
#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <Macros.h>


namespace file_operations
{
template <class T>
void write_vector(const std::string &f_name, int N, T *vec)
{
        std::ofstream f(f_name.c_str(), std::ofstream::out);
        if (!f) throw std::runtime_error("print_vector: error while opening file " + f_name);

        for (int i = 0; i < N; ++i)
        {
            if (!(f << std::setprecision(16) << vec[i] << " " << std::endl))
                throw std::runtime_error("print_vector: error while writing to file " + f_name);
        } 
        f.close();
}


template <class T>
void write_matrix(const std::string &f_name, int Row, int Col, T *matrix)
{
    int N=Col;
    std::ofstream f(f_name.c_str(), std::ofstream::out);
    if (!f) throw std::runtime_error("print_matrix: error while opening file " + f_name);
    for (int i = 0; i<Row; i++)
    {
        for(int j=0;j<Col;j++)
        {
            if(j<Col-1)
                f << std::setprecision(16) << matrix[I2(i,j,Row)] << " ";
            else
                f << matrix[I2(i,j,Row)];

        }
        f << std::endl;
    } 
    
    f.close();
}

int read_matrix_size(const std::string &f_name)
{

    std::ifstream f(f_name.c_str(), std::ifstream::in);
    if (!f) throw std::runtime_error("read_matrix_size: error while opening file " + f_name);
    std::string line;
    int matrix_size=0;
    while (std::getline(f, line)){
        matrix_size++;
    }
    f.close();
    return matrix_size;
}

template <class T>
void read_matrix(const std::string &f_name,  int Row, int Col,  T *matrix){
    std::ifstream f(f_name.c_str(), std::ifstream::in);
    if (!f) throw std::runtime_error("read_matrix: error while opening file " + f_name);
    for (int i = 0; i<Row; i++)
    {
        for(int j=0;j<Col;j++)
        {
            // double val=0;  
            // fscanf(stream, "%le",&val);                
            // matrix[I2(i,j,Row)]=(real)val;
            T val;
            f >> val;
            matrix[I2(i,j,Row)]=(T)val;
        }
        
    } 

    f.close();
}
template <class T>
int read_vector(const std::string &f_name,  int N,  T *vec){

    std::ifstream f(f_name.c_str(), std::ifstream::in);
    if (!f) throw std::runtime_error("read_vector: error while opening file " + f_name);
    for (int i = 0; i<N; i++)
    {
        T val=0;   
        f >> val;             
        vec[i]=(T)val;           
    } 
    f.close();
    return 0;
}


}

#endif