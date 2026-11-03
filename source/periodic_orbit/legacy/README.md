# Archived Periodic-Orbit Implementation

`include/periodic_orbit` preserves the original implementation, `detail`, and
tests without source changes. Tests stay with the original tree so their relative
includes and shared Lorenz/Rossler problem headers remain valid.

This is historical code, not the supported periodic-orbit implementation. It
still depends on legacy timesteppers and, in some examples, archived project
Newton and linear solvers. Preservation does not certify that every old target
builds or runs against current dependencies. Do not change its numerical logic,
test equations, or formatting.

The main Makefile enables `-I source/periodic_orbit/legacy/include` only for
`LEGACY_PERIODIC_ORBIT_TARGETS`, ahead of the normal project include path. This
keeps existing ABC/KF3D applications and the frozen Lorenz timestepper test wired
to this archive without forwarding headers or edits to their sources. Other
historical builds must explicitly opt into the same include root.

New event/section and flow-map code belongs outside `legacy` and uses
`nmfd-timesteppers`. Remove this archive only after its consumers are migrated or
retired and the relevant regressions have been preserved.
