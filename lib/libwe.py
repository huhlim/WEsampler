#!/usr/bin/env python

import os
import sys
import copy
import pickle
import numpy as np

import mdtraj

from simtk.unit import *
from simtk.openmm import *
from simtk.openmm.app import *

from libmpi import *
from libcustom import *

class WE_Walker(object):
    def __init__(self, positions):
        self.id = None
        self.weight = None
        self.state = None
        self._positions = positions
    def __repr__(self):
        return '%6.4f'%self.weight
    def to_mdtraj(self, top):
        r = self.positions.value_in_unit(nanometers)
        unitcell_vectors = self.state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(nanometers)[None,:]
        X = mdtraj.Trajectory([r], top)
        X.unitcell_vectors = unitcell_vectors
        return X
    @property
    def positions(self):
        if self.state is not None:
            self._positions = self.state.getPositions(asNumpy=True)
        return self._positions
    @classmethod
    def initialize_with_runner(cls, runner, positions, **kwarg):
        walker = cls(positions)
        runner.walker_to_simulation(walker, **kwarg)
        runner.simulation_to_walker(walker)
        return walker

class WE_Runner(object):
    def __init__(self, topology, system, integrator, *arg, **kwarg):
        self.topology = topology
        self.system = system
        self.integrator = integrator
        #
        if MPI_RANK > 0:
            random_number_seed = self.integrator.getRandomNumberSeed() + MPI_RANK
            self.integrator.setRandomNumberSeed(random_number_seed)
            #
            PLATFORM = Platform.getPlatformByName('CUDA')
            PROPERTIES = {"CudaDeviceIndex": MPI_GPU_BINDING[MPI_RANK]}
            self.simulation = Simulation(self.topology, self.system, self.integrator, PLATFORM, PROPERTIES, *arg)
        else:
            self.simulation = Simulation(self.topology, self.system, self.integrator)
    def __del__(self):
        self.simulation = None
    def walker_to_simulation(self, walker, **kwarg):
        # copy walker -> simulation
        if walker.state is not None:
            self.simulation.context.setState(walker.state)
        else:
            if 'box' in kwarg:
                self.simulation.context.setPeriodicBoxVectors(*(np.eye(3)*kwarg['box']))
            self.simulation.context.setPositions(walker.positions)
            self.simulation.context.setVelocitiesToTemperature(self.simulation.integrator.getTemperature())
    def simulation_to_walker(self, walker):
        # copy simulation -> walker
        walker.state = self.simulation.context.getState(getPositions=True, getVelocities=True, getEnergy=True, enforcePeriodicBox=True)
    def getTime(self):
        self.time = self.simulation.context.getState().getTime()
        return self.time
    def setTime(self, time=None):
        if time is not None:
            self.time = time
        self.simulation.context.setTime(self.time)
    def propagate(self, walker, n_step):
        self.walker_to_simulation(walker)
        self.setTime()
        #sys.stdout.write("propagating Walker %d, time=%d\n"%(walker.id, self.time.value_in_unit(picosecond)))
        self.simulation.step(n_step)
        self.simulation_to_walker(walker)

class WE_Sampler(object):
    def __init__(self, system, topology, integrator, dist_func, *arg, **kwarg):
        self.system = system
        self.topology = topology
        self.integrator = integrator
        self.dist_func = dist_func
        self.mdtraj_topology = None
        #
        self.p_min = kwarg.get("p_min", 1.0e-12) 
        self.p_max = kwarg.get("p_max", 0.25)
        #
        self.reporters = []
        self.walker_s = []
    def __del__(self):
        pass
    @property
    def n_walker(self):
        return self._n_walker
    def initialize_walker(self, init_s, weight_s=None):
        self.currentStep = 0
        self.walker_s = init_s
        self._n_walker = len(self.walker_s)
        if weight_s is None:
            weight_s = np.ones(self.n_walker, dtype=float) / self.n_walker
        for i,walker in enumerate(self.walker_s):
            walker.id = i
            walker.weight = weight_s[i]
    def define_mdtraj_topology(self, topology):
        self.mdtraj_topology = topology
    def loadCheckpoint(self, fp):
        self.currentStep, self.walker_s = pickle.load(fp)
        self._n_walker = len(self.walker_s)

    def createCheckpoint(self, fout):
        pickle.dump((self.currentStep, self.walker_s), fout)

    def report(self):
        for reporter in self.reporters:
            nextReport = reporter.describeNextReport(self)
            if nextReport[0] > 1: continue
            #
            for walker in self.walker_s:
                reporter.report(self, walker.state)
            #
    def resample(self):
        raise NotImplementedError

def run_sampler_serial(sampler, runner, n_cycle, n_step):
    for i in xrange(n_cycle):
        sampler.currentStep += 1
        runner.getTime()
        for walker in sampler.walker_s:
            runner.propagate(walker, n_step)
        #
        sampler.report()
        sampler.resample()

def run_sampler_mpi(sampler, runner, n_cycle, n_step):
    MPI_SIGNAL_TERMINATE = 0
    MPI_SIGNAL_RUN = 1
    #
    for i in xrange(n_cycle):
        if MPI_RANK == MPI_KING:
            sampler.currentStep += 1
            #
            proc_s = [True for _ in range(1, MPI_SIZE)]
            walker_status = np.array([0 for _ in xrange(sampler.n_walker)], dtype=int)
            while np.any(walker_status != 2):
                for i_proc in range(1, MPI_SIZE):
                    if not proc_s[i_proc-1]: continue
                    for walker in sampler.walker_s:
                        if walker_status[walker.id] == 0:
                            proc_s[i_proc-1] = False
                            MPI_COMM.send(MPI_SIGNAL_RUN, dest=i_proc, tag=10)
                            MPI_COMM.send(walker, dest=i_proc, tag=11)
                            walker_status[walker.id] = 1
                            break
                #
                status = MPI.Status()
                walker = MPI_COMM.recv(source=MPI_ANY_SOURCE, tag=12, status=status)
                i_proc = status.Get_source()
                proc_s[i_proc-1] = True
                #
                sampler.walker_s[walker.id] = walker
                walker_status[walker.id] = 2
            #
            for i_proc in range(1, MPI_SIZE):
                MPI_COMM.send(MPI_SIGNAL_TERMINATE, dest=i_proc, tag=10)
            #
            sampler.report()
            sampler.resample()
        else:
            runner.getTime()
            #
            while True:
                signal = MPI_COMM.recv(source=MPI_KING, tag=10)
                if signal == MPI_SIGNAL_TERMINATE:
                    break
                elif signal == MPI_SIGNAL_RUN:
                    walker = MPI_COMM.recv(source=MPI_KING, tag=11)
                    runner.propagate(walker, n_step)
                    MPI_COMM.send(walker, dest=MPI_KING, tag=12)
        #
        MPI_COMM.barrier()

if MPI_SIZE > 1:
    run_sampler = run_sampler_mpi
else:
    run_sampler = run_sampler_serial

class WExplorer_Sampler(WE_Sampler):
    pass

class REVO_Sampler(WE_Sampler):
    def __init__(self, *arg, **kwarg):
        super(REVO_Sampler, self).__init__(*arg, **kwarg)
        #
        self.d_merge = kwarg.get("d_merge", 1.0)
        self.d_char = kwarg.get("d_char", 1.0)
        self.d_exp = kwarg.get("d_exp", 2.0)
    def get_distance_matrix(self):
        d_mtx = np.zeros((self.n_walker, self.n_walker), dtype=float)
        pdb_s = [w.to_mdtraj(self.mdtraj_topology) for w in self.walker_s]
        for i in xrange(self.n_walker-1):
            for j in xrange(i+1, self.n_walker):
                d_mtx[i,j] = self.dist_func(pdb_s[i], pdb_s[j], self.mdtraj_topology)
        d_mtx += d_mtx.T
        return d_mtx
    def resample(self):
        walker_id0 = [[i] for i in xrange(self.n_walker)]
        weight_s0 = np.array([walker.weight for walker in self.walker_s])
        #
        d_mtx0 = self.get_distance_matrix()
        #
        def calc_walker_info(walker_id, weight_s, d_mtx, i_clone=None, i_merge=None, i_replace=None):
            # if (0,1,2) is given
            #  walker_id: [[0], [1], [2], [3], ...] -> [[0], [1,2], [0], [3], ...]
            #  n_walker_copy: -> [2, 1, 0, 1, ...]
            #  
            walker_id_test = copy.copy(walker_id)
            if i_replace is not None:
                walker_id_test[i_merge] += walker_id_test[i_replace]
                walker_id_test[i_replace] = walker_id_test[i_clone]
            #
            n_walker_copy = np.zeros(self.n_walker, dtype=int)
            for w in walker_id_test:
                n_walker_copy[w[0]] += 1
            #
            weight_s_test = np.zeros_like(weight_s)
            for i,w in enumerate(walker_id_test):
                weight_s_test[i] = sum(weight_s[w]) / n_walker_copy[w[0]]
            #
            d_mtx_test = copy.copy(d_mtx)
            if i_replace is not None:
                d_mtx_test[:,i_replace] = d_mtx_test[:,i_clone]
                d_mtx_test[i_replace,:] = d_mtx_test[i_clone,:]
            return walker_id_test, weight_s_test, n_walker_copy, d_mtx_test
        def calc_novelty(weight_s, p_min=self.p_min):
            novelty = np.log(weight_s) - np.log(p_min)
            return novelty
        def calc_variation(d_mtx, novelty):
            v_mtx = np.power((d_mtx/self.d_char), self.d_exp) * np.outer(novelty, novelty)
            variation = np.sum(v_mtx)
            v_walker = np.sum(v_mtx, axis=0)
            return variation, v_walker
        #
        walker_id, weight_s, n_walker_copy, d_mtx = calc_walker_info(walker_id0, weight_s0, d_mtx0)
        #
        novelty = calc_novelty(weight_s)
        variation, v_walker = calc_variation(d_mtx, novelty)
        #
        while True:
            v_max_index = None      # -> clone
            v_min_index = None      # -> merge
            v_close_index = None    # -> merge
            #
            v_max_candidates = []
            for i,v in enumerate(v_walker):
                if n_walker_copy[i] == 0: continue  # replaced
                if len(walker_id[i]) != 1: continue # merged
                if weight_s[i] / (n_walker_copy[i]+1.0) < self.p_min: continue
                v_max_candidates.append((v, i))
            if len(v_max_candidates) > 0:
                v_max, v_max_index = max(v_max_candidates)
            #
            v_min_candidates = []
            for i,v in enumerate(v_walker):
                if n_walker_copy[i] != 1: continue  # replaced or cloned
                if weight_s[i] > self.p_max: continue
                v_min_candidates.append((v, i))
            if len(v_min_candidates) > 0:
                v_min, v_min_index = min(v_min_candidates)
            #
            v_close_candidates = []
            for i,v in enumerate(v_walker):
                if i in [v_max_index, v_min_index]: continue
                if n_walker_copy[i] != 1: continue  # replaced or cloned
                if weight_s[i]+weight_s[v_min_index] > self.p_max: continue
                if d_mtx[i, v_min_index] > self.d_merge: continue
                v_close_candidates.append((d_mtx[i, v_min_index], i))
            if len(v_close_candidates) > 0:
                v_close, v_close_index = min(v_close_candidates)
            #
            if None in [v_max_index, v_min_index, v_close_index]:
                break
            #
            weight_merge = weight_s[v_min_index] + weight_s[v_close_index]
            v_merge_index = np.random.choice([v_min_index, v_close_index], \
                    p=np.array([weight_s[v_min_index], weight_s[v_close_index]])/weight_merge)
            if v_merge_index == v_min_index:
                v_replace_index = v_close_index
            else:
                v_replace_index = v_min_index
            #
            walker_id_test, weight_s_test, n_walker_copy_test, d_mtx_test = \
                    calc_walker_info(walker_id, weight_s0, d_mtx,
                                     i_clone=v_max_index, i_merge=v_merge_index, i_replace=v_replace_index)
            #
            novelty_test = calc_novelty(weight_s_test)
            variation_test, v_walker_test = calc_variation(d_mtx_test, novelty)
            #
            if variation_test > variation:
                sys.stdout.write("Walker: %d cloned -> %d ; %d + %d merged -> %d\n"%\
                        (v_max_index, v_replace_index, v_merge_index, v_replace_index, v_merge_index))
                walker_id = walker_id_test
                weight_s = weight_s_test
                n_walker_copy = n_walker_copy_test
                d_mtx = d_mtx_test
                variation = variation_test
                v_walker = v_walker_test
            else:
                break
        #
        walker_s_updated = []
        for i in range(self.n_walker):
            w = copy.deepcopy(self.walker_s[walker_id[i][0]])
            w.weight = weight_s[i]
            walker_s_updated.append(w)
        #
        sys.stdout.write("Walker: %s\n"%(' '.join(['%6.4f'%w.weight for w in walker_s_updated])))

        self.walker_s = walker_s_updated
        for i,walker in enumerate(self.walker_s):
            walker.id = i

