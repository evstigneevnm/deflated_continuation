#ifndef __TIME_STEPPER_GENERIC_TIME_STEP_H__
#define __TIME_STEPPER_GENERIC_TIME_STEP_H__


#include <memory>
#include <string>
#include <utility>
#include "detail/all_methods_enum.h"
#include "detail/butcher_tables.h"

namespace time_steppers
{

template<class VectorOperations>
class generic_time_step
{
public:
    using T = typename VectorOperations::scalar_type;
    using T_vec = typename VectorOperations::vector_type;
    using method_type = detail::methods;
    using table_type = time_steppers::detail::tableu;

    generic_time_step() {};
    virtual ~generic_time_step() = default;

    virtual void scheme(const std::string& method_p) = 0;

    virtual void set_parameter(const T param_p) = 0;

    virtual T get_dt()const = 0;

    virtual T get_time()const = 0;

    virtual void force_dt_single_step(const T dt_p) = 0;

    virtual void pre_execte_step() = 0;

    virtual std::size_t get_iteration()const = 0;

    virtual void init_steps(const T_vec& in_p) = 0;

    virtual bool check_reject_step()const = 0;

    virtual void set_initial_time_interval(const std::pair<T,T> time_interval_p) = 0;

    virtual void force_undo_step() = 0;

    virtual void reset_steps() = 0;

    virtual bool execute_forced_dt(const T dt_forced_p, const T_vec& in_p, T_vec& out_p) = 0;

    virtual bool execute(const T_vec& in_p, T_vec& out_p) = 0;

};

}

#endif