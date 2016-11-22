import numpy as np
import floq.core.evolution as ev


def evolve_system(system):
    return ev.get_u(system.hf, system.params)


def evolve_system_with_derivatives(system):
    return ev.get_u_and_du(system.hf, system.dhf, system.params)
