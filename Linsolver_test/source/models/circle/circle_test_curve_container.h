#ifndef __CIRCLE_TEST_CURVE_CONTAINER_H__
#define __CIRCLE_TEST_CURVE_CONTAINER_H__


//class VectorOperations, class Loggin, class NewtonMethod, class NonlinearOperator, class Knots
 
typedef knots<real> knots_t;
typedef curve_storage<vec_ops_real, log_t, newton_t, circle_t, knots_t> curve_storage_t;



#endif // __CIRCLE_TEST_CURVE_CONTAINER_H__