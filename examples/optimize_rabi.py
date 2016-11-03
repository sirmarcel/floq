import sys
import numpy as np
sys.path.append('..')
import floq.systems.rabi as rb
import floq.evolution as ev
import floq.optimization_task as ot
import floq.optimizer as opt
import floq.fidelity as fid
import pdb

s = rb.get_rabi_system([1.2,2.8],5.0)
print ev.evolve_system(s.get_system([0.5],1.0))

target = np.array([[0.0374008 - 0.883626j, 0.223747 - 0.409566j], [0.456211 + 0.0983779j, -0.693177 - 0.549271j]])
init = np.array([0.1])

task = ot.OptimizationTaskWithFunctions(s,fid.overlap_to_minimise,fid.d_overlap_to_minimise,target,0.99,init)


optimizer = opt.SciPyOptimizer(task,1.0)
# 

res = optimizer.optimize()
print res