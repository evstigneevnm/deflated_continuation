#ifndef __CURVE_HELPER_CONTAINER_H__
#define __CURVE_HELPER_CONTAINER_H__



namespace container
{

template<class VectorOperations>
class curve_helper_container
{
public:    
    typedef typename VectorOperations::vector_type  T_vec;

    curve_helper_container(VectorOperations*& vec_ops_):
    vec_ops(vec_ops_)
    {
        vec_ops->init_vector(x0); vec_ops->start_use_vector(x0);
        vec_ops->init_vector(x1); vec_ops->start_use_vector(x1);    
    }
    ~curve_helper_container()
    {
        vec_ops->stop_use_vector(x0); vec_ops->free_vector(x0);
        vec_ops->stop_use_vector(x1); vec_ops->free_vector(x1);        
    }

    void get_refs(T_vec& x0_, T_vec& x1_)
    {
        x0_ = x0;
        x1_ = x1;
    }
private:
    T_vec x0 = nullptr;
    T_vec x1 = nullptr;
    VectorOperations* vec_ops;


};

}

#endif // __CURVE_HELPER_CONTAINER_H__