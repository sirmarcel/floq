import sys
sys.path.append('..')
import numpy as np
import floq.core.spin as spin
import floq.core.fixed_system as fs
import timeit


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped


def time_u(get_u, hf, params):
    time = min(timeit.Timer(wrapper(get_u, hf, params)).repeat(5, 20))/20
    return " U: " + str(round(time*1000, 3)) + " ms per execution"


def time_du(func, hf, dhf, params):
    time = min(timeit.Timer(wrapper(func, hf, dhf, params)).repeat(3, 1))
    return "dU: " + str(round(time, 3)) + " s per execution"


ncomp = 6
freq = 1.1
controls = 0.5*np.ones(2*ncomp)

hf = spin.hf(ncomp, freq, controls)
dhf = spin.dhf(ncomp, 1.0)
system = fs.FixedSystem(hf, dhf, 101, 1.0, 1.0)

params = system.params

print "---- Current version (Numba)"
import floq.core.evolution as ev

print time_u(ev.get_u, hf, params)
print time_du(ev.get_u_and_du, hf, dhf, params)


print "---- Better algorithm for dU"
import floq.museum.p4.evolution as ev

print time_u(ev.get_u, hf, params)
print time_du(ev.get_u_and_du, hf, dhf, params)


print "---- Use less Python"
import floq.museum.p3.evolution as ev

print time_u(ev.get_u, hf, params)
print time_du(ev.get_u_and_du, hf, dhf, params)


print "---- Use sparse matrix library"
import floq.museum.p2.evolution as ev

print time_u(ev.get_u, hf, params)
print time_du(ev.get_u_and_du, hf, dhf, params)
print "---- Baseline"
import floq.museum.p1.evolution as ev

print time_u(ev.get_u, hf, params)
print time_du(ev.get_u_and_du, hf, dhf, params)