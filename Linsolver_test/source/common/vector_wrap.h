#ifndef __VECTOR_WRAP_H__
#define __VECTOR_WRAP_H__


template <class VecOps>
class vector_wrap
{
public: 
    typedef VecOps vector_operations;
    typedef typename VecOps::vector_type  vector_type;
    typedef typename VecOps::scalar_type  scalar_type;

private:
    typedef scalar_type T;
    typedef vector_type T_vec;


    VecOps* vec_ops;
    bool allocated = false;
    void set_op(VecOps* vec_ops_){ vec_ops = vec_ops_; }

public:
    vector_wrap()
    {
    }
    ~vector_wrap()
    {
        free();
    }

    void alloc(VecOps* vec_ops_)
    {
        set_op(vec_ops_);

        if(!allocated)
        {
            vec_ops->init_vector(x); vec_ops->start_use_vector(x); 
            allocated = true;
        }
    }
    void free()
    {
        
        if(allocated)
        {
            vec_ops->stop_use_vector(x); vec_ops->free_vector(x);
            allocated = false;
        }
    }

    T_vec& get_ref()
    {
        return(x);
    }

    T_vec x = nullptr;

};


#endif // __VECTOR_WRAP_H__