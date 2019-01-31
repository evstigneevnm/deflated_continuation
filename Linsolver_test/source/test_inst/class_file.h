#pragma once

#include <cstddef>

template<typename T, int BLOCK_SIZE = 256>
class class_file
{
public:
    typedef T  scalar_type;
    typedef T* vector_type;
    
    class_file(size_t sz_): sz(sz_)
    {

    }
 
    ~class_file()
    {

    }
    void start_use_vector(T*& x);
    void stop_use_vector(T*& x);
    void add_vectors(const T*& x, T*& y);

private:
    size_t sz;

};