#pragma once

#include <cstddef>
#include <cuda_runtime.h>
#include <utils/cuda_support.h>



namespace test_class
{

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
    

    void start_use_vector(T*& x)
    {
        x=device_allocate<T>(sz);
    }

    void stop_use_vector(T*& x)
    {
        cudaFree(x);
    }

    void add_vectors(const T*& x, T*& y);
    
    bool is_vector(const T*& x) const
    {
        bool result = false;
        if(x!=NULL)
            result=true;
        return result;
    }


private:
    size_t sz;


};

}

