import numpy as np
import floq.core.evolution as ev


def evolve_system(system):
    return ev.do_evolution(system.hf, system.params)


def evolve_system_with_derivatives(system):
    return ev.do_evolution_with_derivatives(system.hf, system.dhf, system.params)
