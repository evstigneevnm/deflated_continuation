#include <vector>
#include "test_type_traits.hpp"


template<class Operator, class Vector>
void apply_operator(Operator* op, Vector* vec)
{

    op->F(vec);
    if constexpr(Operator::is_periodic_orbit_reprojected::value)
    {
        op->reproject(vec);
    }

}


int main(int argc, char const *argv[])
{
    using vec_t = std::vector<double>;
    using op1_t = nonlinear_operator_test_1<vec_t>;
    using op2_t = nonlinear_operator_test_2<vec_t>;
    vec_t vec(10, 0);

    op1_t op1;
    op2_t op2;

    apply_operator<op1_t, vec_t>(&op1, &vec);
    apply_operator<op2_t, vec_t>(&op2, &vec);

    return 0;
}