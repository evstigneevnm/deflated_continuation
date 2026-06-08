#ifndef __TIME_STEPPER_POSITIVE_PRESERVING_DUMMY_H__
#define __TIME_STEPPER_POSITIVE_PRESERVING_DUMMY_H__

namespace time_steppers
{
namespace detail
{

template<class VectorOperations>
class positive_preserving_dummy
{
public:
	using T_vec = typename VectorOperations::vector_type;

	positive_preserving_dummy() = default;
	~positive_preserving_dummy() = default;

	bool apply(const T_vec& vec_2_check_p)const 
	//this can also modify the vector if desired, but in here we assume that it is constant
	{
		// vector to be checked for components
		return true;
	}

	
};


}
}


#endif