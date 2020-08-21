#ifndef __POINTER_QUEUE_H__
#define __POINTER_QUEUE_H__

#include <cassert>
#include <map>
#include <deque>
#include <utils/cuda_support.h>



namespace utils
{

template<typename T>
class cuda_allocator
{
private:
    typedef std::pair<bool, T*> value_t; 
    typedef std::map<uint64_t, value_t > container_t;

    uint64_t ref2int(const T* vec_)
    {
        return(reinterpret_cast<uint64_t>(vec_));
    }
    
    int m_allocations = 0;    
    int m_available = 0;
    std::size_t max_vectors = 0;
    std::size_t vector_size = 0;
    container_t container;

public:
    cuda_allocator(std::size_t vector_size_, std::size_t max_vectors_):
    vector_size(vector_size_),
    max_vectors(max_vectors_)
    {
        m_available = max_vectors;
        for(int j=0;j<max_vectors;j++)
        {
            T* mem = device_allocate<T>(vector_size);
            uint64_t ref_int = ref2int(mem);
            container.insert( {ref_int, value_t(false, mem)} );
        }
    }
    ~cuda_allocator() 
    {
        
        for(auto &x: container)
        {
            device_deallocate<T>(x.second.second);
        }
        
        assert((m_allocations == 0) && "destructor in cuda_allocator has invalid value of number of allocation.");
        
    }

    T* take(const T* ptr_) 
    {
        
        T* ret = nullptr;
        for(auto &x: container)
        {
            if(!x.second.first)
            {
                x.second.first = true;
                ret = x.second.second;
                device_2_device_cpy<T>(ptr_, ret, vector_size);
                break;
            }
        }
        if(ret == nullptr)
        {
            throw std::runtime_error(std::string("vector_pool: no more vectors in the pool!") );
        }

        m_available--;
        ++m_allocations;
        return ret;
    }   

    void release(T* val_) 
    {
        uint64_t key = ref2int(val_);
        try
        {
            container.at(key).first = false;
            --m_allocations;
            m_available++;
        }
        catch(const std::exception& e)
        {
            
            throw std::runtime_error(std::string("vector_pool: release: ") + e.what() );
        }        

    }




};



template<class T>
class pointer_queue
{
private:
    typedef cuda_allocator<T> allocator_t;
    size_t max_elements;
    allocator_t* allocator;
    std::deque<T*> container;

public: 
    pointer_queue(std::size_t vec_size_, std::size_t max_elements_):
    max_elements(max_elements_)
    {
        allocator = new allocator_t(vec_size_, max_elements);
        //container.reserve(max_elements);
    }
    ~pointer_queue()
    {
        clear();
        delete allocator;
    }

    void clear()
    {
        for(auto &x: container)
        {
            allocator->release(x); 
        }   
        container.clear();
    }

    T* at(int j)
    {
        return(container.at(j));
    }
    
    bool is_queue_filled()
    {
        if(container.size() == max_elements)
            return true;
        else
            return false;
    }

    void push(const T* x)
    {
        if( container.size() == max_elements)
        {
            T* ref_front = container.front();
            allocator->release(ref_front);
            container.pop_front();
        }
        T* ref_back = allocator->take(x);
        container.push_back(ref_back);
    }

};



}


#endif // __POINTER_QUEUE_H__