#include <scfd/utils/log_std.h>
#include <common/scfd_serial_cpu_vector_operations.h>
#include "../detail/glued_nonlinear_operator_and_jacobian.h"
#include "rossler_operator.h"


int main(int argc, char const *argv[])
{
    using log_t = scfd::utils::log_std;
    using real = SCALAR_TYPE;
    using T = real;
    using vec_ops_t = scfd_serial_cpu_vector_operations<real>;
    using vec_t = typename vec_ops_t::vector_type;
    using nlin_op_t = nonlinear_operators::rossler<vec_ops_t>;
    using glued_nonlin_op_t = periodic_orbit::detail::glued_nonlinear_operator_and_jacobian<vec_ops_t, nlin_op_t>;
    using glued_vec_t = typename glued_nonlin_op_t::vector_type;
    
    T ref_error = std::numeric_limits<T>::epsilon();

    if(argc == 2)
    {
        std::string scheme_name(argv[1]);
    }

    log_t log;
    log.info("test glued nonlinear operator.");
    vec_ops_t vec_ops(3);
    nlin_op_t rossler(&vec_ops, 2, 0.2, 0.2, 5.7);

    glued_nonlin_op_t glued_nonlin_op(&vec_ops, &rossler);
    glued_vec_t x,v;
    glued_nonlin_op.glued_vector_operations()->init_vector(x);
    glued_nonlin_op.glued_vector_operations()->start_use_vector(x);
    glued_nonlin_op.glued_vector_operations()->init_vector(v);
    glued_nonlin_op.glued_vector_operations()->start_use_vector(v);

    glued_nonlin_op.glued_vector_operations()->assign_scalar(0.1,x);
    glued_nonlin_op.glued_vector_operations()->assign_scalar(0.5,v);

    glued_nonlin_op.F(0.0, x, 5.7, v);

    // std::cout << v.comp(0)[0] << " " << v.comp(0)[1] << " " << v.comp(0)[2] << std::endl;
    // std::cout << v.comp(1)[0] << " " << v.comp(1)[1] << " " << v.comp(1)[2] << std::endl;

    auto v_norm = glued_nonlin_op.glued_vector_operations()->norm(v);
    decltype(v_norm) ref_val = std::sqrt(0.5409);
    auto err = std::abs(v_norm - ref_val )/ref_val;
    log.info_f("returned vector norm = %le, error = %le ", v_norm, err );

    glued_nonlin_op.glued_vector_operations()->stop_use_vector(x);
    glued_nonlin_op.glued_vector_operations()->free_vector(x);
    glued_nonlin_op.glued_vector_operations()->stop_use_vector(v);
    glued_nonlin_op.glued_vector_operations()->free_vector(v);

    int N_tests = 1;
    const uint8_t N = 3;
    auto ref_error_val = ref_error*N_tests*std::sqrt(N);
    if(err > ref_error*N_tests*std::sqrt(N) )
    {
        log.error_f("Got error = %e with reference = %e", err,  ref_error_val);
        return 1;
    }
    else
    {
        log.info_f("No errors with reference = %e", ref_error_val);
        return 0;    
    }


	return 0;
}
