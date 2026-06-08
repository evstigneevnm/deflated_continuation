#ifndef __STABILITY_IRAM_CPU_ARRAYS_H__
#define __STABILITY_IRAM_CPU_ARRAYS_H__

#include<vector>
#include<stdexcept>

template<class T>
class cpu_arrays
{
public:
    cpu_arrays()
    {
    }
    cpu_arrays(size_t sz_):
    sz(sz_)
    {
        container = std::vector<T>(sz, 0);
        is_init = true;
    }
    ~cpu_arrays()
    {
    }

    void init(size_t sz_)
    {
        if(!is_init)
        {
            container = std::vector<T>(sz, 0);
            is_init = true;        
        }
        else
        {
            throw std::runtime_error("attempt to init already initialized array.");
        }
    }
    

    std::vector<T> container;

private:
    bool is_init = false;
    size_t sz;
};



#endif