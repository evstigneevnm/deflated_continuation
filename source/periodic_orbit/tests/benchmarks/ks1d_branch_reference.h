#ifndef PERIODIC_ORBIT_TESTS_KS1D_BRANCH_REFERENCE_H
#define PERIODIC_ORBIT_TESTS_KS1D_BRANCH_REFERENCE_H
#include "ks1d_orbit_support.h"

namespace ks1d_orbit
{
constexpr double p0 = 68., lower = 67.999, upper = 68.001;
// Independent step-level RK45 benchmark, M=1, common-parameter probes.
// Compare the lifted Fourier state, not section coordinates or just an amplitude.
struct reference_point
{
    double                          parameter, period;
    std::array<double, seed.size()> coefficients;
};
constexpr reference_point reference[] = {
    { lower,
      .01515307799761766,
      { 3.4673219313511034, -3.025633467841686, 52.9335843425138, 4.731334841560447, -3.2241249956955165,
        15.822294020637356, 1.7723762302340096, -.7813868478525984, 1.8946804477848374, .36597388838684525,
        -.1942978551899088 } },
    { upper,
      .01515430221001677,
      { 3.475290144782186, -3.0283605619025598, 52.93172198242308, 4.730121294323739, -3.2244738085765574,
        15.821773615337458, 1.7722760988546171, -.7815278436413722, 1.8946036226157676, .3659522709150246,
        -.19430695839266474 } }
};
}
#endif
