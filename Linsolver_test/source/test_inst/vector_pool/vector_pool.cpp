// Example program
#include <map>
#include <string>
#include <iostream>
//#include <common/gpu_vector_opearations.h>
//#include <common/vector_wrap.h>


double* d_alloc(size_t sz)
{
    double* ret = nullptr;
    ret = (double*)malloc(sz*sizeof(double));
    return(ret);
}

void set_vec(double* vec_, double val, size_t N)
{
    for(int j=0;j<N;j++)
    {
        vec_[j] = val;
    }
}


uint64_t ref2int(const double* vec_)
{
    return(reinterpret_cast<uint64_t>(vec_));
}

template<class Container>
double* take(Container& container)
{
    double *ret = nullptr;
    for(auto &x: container)
    {
        if(!x.second.first)
        {
            x.second.first = true;
            ret = x.second.second;
            break;
        }
    }
    if(ret == nullptr)
    {
        throw std::runtime_error(std::string("vector_pool: no more vectors in the pool!") );
    }
    
    return(ret);
}

template<class Value, class Container>
void release(double* val_, Container& container)
{
   container[ref2int(val_)].first = false;
}

int main()
{
    typedef std::pair<bool, double*> value_t;
    
    typedef std::map<uint64_t, value_t> container_t;
    
    container_t container;
    size_t N = 100000000;
    double *v1 = d_alloc(N);
    set_vec(v1, 1, N);
    double *v2 = d_alloc(N);
    set_vec(v2, 2, N);
    double *v3 = d_alloc(N);
    set_vec(v3, 3, N);
    double *v4 = d_alloc(N);
    set_vec(v4, 4, N);
    
    container.insert( {ref2int(v1), value_t(false, v1)} );
    container.insert( {ref2int(v2), value_t(false, v2)} );
    container.insert( {ref2int(v3), value_t(false, v3)} );
    container.insert( {ref2int(v4), value_t(false, v4)} );
    
    for(auto &x: container)
    {
        std::cout << x.first << ": " << x.second.second[99] << std::endl;
    }
  
    for(int j=0;j<500000;j++)
    {
        double* t1 = take<container_t>(container);
        std::cout << "t1 took:" << ref2int(t1) << std::endl;
        double* t2 = take<container_t>(container);
        std::cout << "t2 took:" << ref2int(t2) << std::endl;
        
        release<value_t, container_t>(t1, container);
        std::cout << "t1 released:" << ref2int(t1) << std::endl;
        
        double* t3 = take<container_t>(container);
        std::cout << "t3 took:" << ref2int(t3) << std::endl;
        
        release<value_t, container_t>(t2, container);
        std::cout << "t2 released:" << ref2int(t2) << std::endl;        
        //!!commenting these causes planned throw!!
        release<value_t, container_t>(t3, container);
        std::cout << "t3 released:" << ref2int(t3) << std::endl;        
    }
    
  
    for(auto &x: container)
    {
        free(x.second.second);
    }
}