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
        std::cout << "class_file constructor" << std::endl;
        NC= new nested_class(*this);
        
    }
 
    ~class_file()
    {
        std::cout << "class_test distructor" << std::endl;
        delete NC;
    }
    
    class nested_class
    {
    public:
        nested_class(class_file& cf_): cf(cf_)
        {
            std::cout << "nested constructor" << std::endl;
            nested_size=cf.sz;
            std::cout << "  set nested class size to " << nested_size << std::endl;
            vector=NULL;
            vector=(T*) malloc(sizeof(T)*nested_size);
            std::cout << "  vector created" << std::endl;

        }

        ~nested_class()
        {
            std::cout << "nested distructor" << std::endl;
            if(vector!=NULL){
                free(vector);
                std::cout << "  vector removed" << std::endl;
            }

        }

        void hi()
        {
            cf.sz=1223;
            std::cout << "  hi(): class_file::sz changed to " << cf.sz << std::endl;
        }
    private:
        size_t nested_size;
        class_file& cf;
        T *vector=NULL;
        
    };


 
    size_t get_size()
    {
        return sz;
    }

    void start_use_vector(T*& x)
    {
        std::cout << "  class_test sz==" << sz << std::endl;
        x=device_allocate<T>(sz);
        NC->hi();
        std::cout << "  class_test sz==" << sz << std::endl;
        sz=100000;
        std::cout << "  setting class_test sz=" << sz << std::endl;
        
    }

    void stop_use_vector(T*& x)
    {
        cudaFree(x);
    }

  
    bool is_vector(const T*& x) const
    {
        bool result = false;
        if(x!=NULL)
            result=true;
        return result;
    }


private:
    size_t sz;
    nested_class *NC;


};

}

