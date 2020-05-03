// Example program
#include <map>
#include <string>
#include <iostream>

#include <external_libraries/cublas_wrap.h>
#include <common/gpu_vector_operations.h>
#include <common/vector_wrap.h>


template<class T>
uint64_t ref2int(const T* vec_)
{
    return(reinterpret_cast<uint64_t>(vec_));
}

template<class Vec,  class Container>
Vec* take(Container& container)
{
    Vec *ret = nullptr;
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

template<class Vec, class Container>
void release(Vec *val_, Container& container)
{
   uint64_t key = ref2int(val_);
   container[key].first = false;
}


int main(int argc, char const *argv[])
{
 
    typedef double T;
    typedef gpu_vector_operations<T> vec_ops_t;
    typedef typename vec_ops_t::vector_type T_vec;
    typedef vector_wrap<vec_ops_t> vec_t;


    typedef std::pair<bool, vec_t*> value_t;
    typedef std::map<uint64_t, value_t> container_t;
    
    container_t container;
    size_t N = 1000;

    init_cuda(-1);
    cublas_wrap CUBLAS;
    
    CUBLAS.set_pointer_location_device(false);

    vec_ops_t vec_ops(N, &CUBLAS);    
    

    
    for(int j=0;j<10;j++)
    {
        vec_t *v1 = new vec_t();
        container.insert( {ref2int<vec_t>(v1), value_t(false, v1 ) } );
    }

    for(auto &x: container)
    {   
        x.second.second->alloc(&vec_ops);     
        vec_ops.assign_scalar(1.0, x.second.second->x );
    }
    

    for(int j=0;j<100;j++)
    {
        vec_t* t1 = take<vec_t, container_t>(container);
        vec_t* t2 = take<vec_t, container_t>(container);
        
        release<vec_t, container_t>(t1, container);
        vec_t* t3 = take<vec_t, container_t>(container);
        vec_ops.assign_scalar(10.0, t2->x );
        vec_ops.assign_scalar(100.0, t3->x );

//        std::cout << t1.x << std::endl;
        release<vec_t, container_t>(t2, container);
        release<vec_t, container_t>(t3, container);

        // std::cout << "t1 took:" << ref2int(t1) << std::endl;
        // double* t2 = take<container_t>(container);
        // std::cout << "t2 took:" << ref2int(t2) << std::endl;
        
        // release<value_t, container_t>(t1, container);
        // std::cout << "t1 released:" << ref2int(t1) << std::endl;
        
        // double* t3 = take<container_t>(container);
        // std::cout << "t3 took:" << ref2int(t3) << std::endl;
        
        // release<value_t, container_t>(t2, container);
        // std::cout << "t2 released:" << ref2int(t2) << std::endl;        
        // //!!commenting these causes planned throw!!
        // release<value_t, container_t>(t3, container);
        // std::cout << "t3 released:" << ref2int(t3) << std::endl;        
    }

    std::cout << "=--=" << std::endl;
    for(auto &x: container)
    {
        std::cout << vec_ops.norm(x.second.second->x) << std::endl;
        delete x.second.second;
    }

    return(0);
}