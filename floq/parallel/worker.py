# You got nothing to lose but your chains!
import multiprocessing as mp
import numpy as np
from floq.optimization.fidelity import FidelityComputerBase


class FidelityWorker(mp.Process):
    def __init__(self, fids, pipe_in, pipe_out):
        super(FidelityWorker, self).__init__()
        self.pipe_in = pipe_in
        self.pipe_out = pipe_out
        self.fids = fids
        print 'Worker initialised with ' + str(len(self.fids)) + ' fidelities'


    def run(self):
        while True:
            msg = self.pipe_in.recv()
            if msg is not None:
                instruction, ctrl = msg[0], msg[1]
                # print msg
                if instruction == 'f':
                    f = np.sum([fid.f(ctrl) for fid in self.fids])
                    self.pipe_out.send(f)
                else:
                    df = np.sum([fid.df(ctrl) for fid in self.fids], axis=0)
                    self.pipe_out.send(df)
            else:
                break



class FidelityMaster(FidelityComputerBase):
    """
    With a given Ensemble, and a FidelityComputer,
    calculate the average fidelity over the whole ensemble.
    """

    def __init__(self, nworker, ensemble, fidelity, **params):
        self.fidelities = [fidelity(sys, **params) for sys in ensemble.systems]
        self.n = len(ensemble.systems)
        self.nworker = nworker

        # chunk_size = self.n/nworker + self.n % nworker  # divide and round up, with integers
        # print chunk_size
        self.fidelities_chunked = chunks(self.fidelities, nworker)
        self._make_workers()


    def _make_workers(self):
        in_pipes = [mp.Pipe() for i in xrange(self.nworker)]
        out_pipes = [mp.Pipe() for i in xrange(self.nworker)]

        self.workers = []
        for i in xrange(self.nworker):
            worker = FidelityWorker(self.fidelities_chunked[i], in_pipes[i][1], out_pipes[i][1])
            worker.start()
            self.workers.append(worker)

        self.ins = [in_pipes[i][0] for i in xrange(self.nworker)]
        self.outs = [out_pipes[i][0] for i in xrange(self.nworker)]


    def f(self, controls_and_t):
        for pipe in self.ins:
            pipe.send(['f', controls_and_t])

        tmp = []
        for pipe in self.outs:
            tmp.append(pipe.recv())

        return np.sum(tmp)/self.n


    def df(self, controls_and_t):
        for pipe in self.ins:
            pipe.send(['df', controls_and_t])

        tmp = []
        for pipe in self.outs:
            tmp.append(pipe.recv())

        return np.sum(tmp, axis=0)/self.n


    def kill(self):
        for worker in self.workers:
            worker.terminate()



def chunks(l, n):
    """
    Split list l into n-sized chunks.
    """
    # n = max(1, n)
    # return [l[i:i+n] for i in xrange(0, len(l), n)]

    k, m = divmod(len(l), n)
    return list((l[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n)))
