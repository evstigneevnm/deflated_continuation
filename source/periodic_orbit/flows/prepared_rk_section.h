#ifndef PERIODIC_ORBIT_PREPARED_RK_SECTION_H
#define PERIODIC_ORBIT_PREPARED_RK_SECTION_H

#include <periodic_orbit/flows/prepared_section.h>

namespace periodic_orbit
{
template <class Space, class Replay, class Section>
using prepared_rk_section = prepared_section<Space, Replay, Section>;
}
#endif
