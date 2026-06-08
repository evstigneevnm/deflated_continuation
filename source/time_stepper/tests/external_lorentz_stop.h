#ifndef __TIME_STEPPER_EXTERNAL_LORENZ_STOP_H__
#define __TIME_STEPPER_EXTERNAL_LORENZ_STOP_H__


namespace time_steppers
{
namespace detail
{

template <class VectorOperations, class SingleStepper>
struct external_lorentz_stop
{
    using T     = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;

    external_lorentz_stop( VectorOperations *vec_ops ) : vec_ops_( vec_ops )
    {
        vec_ops_->init_vector( dv_ );
        vec_ops_->start_use_vector( dv_ );
    }
    ~external_lorentz_stop()
    {
        vec_ops_->stop_use_vector( dv_ );
        vec_ops_->free_vector( dv_ );
    }

    bool apply( T simulated_time, T &dt, T_vec &v_in, T_vec &v_out )
    {
        //scalar_type mul_x, const vector_type& x, scalar_type mul_y, const vector_type& y, vector_type& z)
        vec_ops_->assign_mul( -1, v_in, 1, v_out, dv_ );
        if ( vec_ops_->norm_l2( dv_ ) < 1.0e-5 )
        {
            std::cout << "### triggered_stop! ###" << std::endl;
            return true;
        }
        else
            return false; //returns external control of the finish flag
    }

private:
    VectorOperations *vec_ops_;
    T_vec             dv_;
};


}
}


#endif