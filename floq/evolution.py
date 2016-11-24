import numpy as np
import floq.core.evolution as ev
import floq.errors as er
import floq.helpers as h


def evolve_system(system):
    u, vals, vecs, phi, psi, nz = evolve_system_and_increase_nz(system)
    return evolve_system_and_increase_nz(system)[0]


def evolve_system_with_derivatives(system):
    u, vals, vecs, phi, psi, nz = evolve_system_and_increase_nz(system)
    du = ev.get_du_from_eigensystem(system.dhf, psi, vals, vecs, system.params)
    return [u, du, nz]


def evolve_system_and_increase_nz(system):
    # Increase nz until U can be computed,
    # then return U and intermediary results,
    # which take the form [u, vals, vecs, phi, psi]
    [nz_okay, results] = test_nz(system)
    while nz_okay is False:
        system.params.nz += 2
        [nz_okay, results] = test_nz(system)
    return results + [system.params.nz]


def test_nz(system):
        # Try to compute U for the given system,
        # if an error occurs or U is not unitary
        # return False, else return [u, vecs, vals, phi, psi]
        try:
            results = ev.get_u_and_eigensystem(system.hf, system.params)
            if h.is_unitary(results[0]):
                return [True, results]
            else:
                return [False, []]
        except er.EigenvalueNumberError:
            return [False, []]
