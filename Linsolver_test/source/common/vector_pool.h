#ifndef __VECTOR_POOL_H__
#define __VECTOR_POOL_H__


#include <map>
#include <exception>
#include <string>

template<class VectorType>
class vector_pool
{
private:

    typedef VectorType vec_t;
    typedef typename vec_t::vector_operations vec_ops_t;

    typedef std::pair<bool, vec_t*> value_t;  
    typedef std::map<uint64_t, value_t> container_t;

    container_t container; //main map container

    uint64_t ref2int(const VectorType* vec_)
    {
        return(reinterpret_cast<uint64_t>(vec_));
    }

public:
    vector_pool()
    {
    }
    ~vector_pool()
    {
        free_all();
    }

    void alloc_all(vec_ops_t* vec_ops, size_t pool_size_)
    {
        if(container.empty())
        {
            try
            {
                for(unsigned int j=0;j<pool_size_; j++)
                {
                    vec_t *v1 = new vec_t();
                    container.insert( {ref2int(v1), value_t(false, v1) } );            
                }
                for(auto &x: container)
                {   
                    x.second.second->alloc(vec_ops);     
                }
            }
            catch(const std::exception& e)
            {
                throw std::runtime_error(std::string("vector_pool: alloc_all: ") + e.what()) ;
            }
        }
    }   
    void free_all()
    {
        if(!container.empty())
        {
            for(auto &x: container)
            {
                delete x.second.second;
            }
            container.clear();
        }
    }
    
    vec_t* take()
    {
        vec_t *ret = nullptr;
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
    void release(vec_t *val_)
    {
        uint64_t key = ref2int(val_);
        try
        {
            container.at(key).first = false;
        }
        catch(const std::exception& e)
        {
            
            throw std::runtime_error(std::string("vector_pool: release: ") + e.what() );
        }
    }



};



#endif // __VECTOR_POOL_H__