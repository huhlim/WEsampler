#!/usr/bin/env python

import os
import sys
import glob
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

import mdtraj
import numpy as np

from simtk.unit import *
from simtk.openmm import *
from simtk.openmm.app import *

sys.path.insert(0, '%s/../lib'%os.path.dirname(os.path.abspath(__file__)))
from libwe import REVO_Sampler, WE_Walker, WE_Runner, run_sampler
from libcustom import read_custom_restraint, construct_custom_restraint

class Restrained_WE_Walker(WE_Walker):
    def __init__(self, positions):
        super(Restrained_WE_Walker, self).__init__(positions)
        self.restraint_s = []
    def define_restraint(self, restraint_s):
        self.restraint_s = restraint_s
    def update_restraint(self, top):
        pdb = self.to_mdtraj(top)
        #
        for force_index, force_info, force_data in self.restraint_s:
            force_type = force_info.split("_")[0]
            use_flatbottom = ('flat' in force_info)
            #
            if force_type == 'pos':
                for i in xrange(force_data.getNumParticles()):
                    atom, p0 = force_data.getParticleParameters(i)
                    p1 = copy.deepcopy(list(p0))
                    for i in xrange(3):
                        p1[i+1] = pdb.xyz[0, atom][i]
                    force_data.setParticleParameters(i, atom, p1)
            elif force_type == 'bond':
                for i in xrange(force_data.getNumBonds()):
                    atom_1, atom_2, p0 = force_data.getBondParameters(i)
                    p1 = copy.deepcopy(list(p0))
                    p1[1] = mdtraj.compute_distances(pdb, [(atom_1, atom_2)])[0] * nanometers
                    force_data.setBondParameters(i, atom_1, atom_2, p1)

    def feed_restraint(self, simulation):
        for force_index, force_info, force_data in self.restraint_s:
            force_type = force_info.split("_")[0]
            #
            simulation_force = simulation.system.getForce(force_index)
            #
            if force_type == 'pos':
                for i in xrange(simulation_force.getNumParticles()):
                    atom = simulation_force.getParticleParameters(i)[0]
                    parameters = force_data.getParticleParameters(i)[1]
                    simulation_force.setParticleParameters(i, atom, parameters)
            elif force_type == 'bond':
                for i in xrange(simulation_force.getNumBonds()):
                    atom_1, atom_2, _ = simulation_force.getBondParameters(i)
                    parameters = force_data.getBondParameters(i)[2]
                    simulation_force.setBondParameters(i, atom_1, atom_2, parameters)
            #
            simulation_force.updateParametersInContext(simulation.context)

class Restrained_WE_Runner(WE_Runner):
    def walker_to_simulation(self, walker, **kwarg):
        super(Restrained_WE_Runner, self).walker_to_simulation(walker, **kwarg)
        walker.feed_restraint(self.simulation)

def calpha_rmsd(x,y, top):
    calphaIndex = top.select("name CA")
    return mdtraj.rmsd(x, y, atom_indices=calphaIndex)[0] * 10.0

def run(arg):
    t_init = time.time()
    n_init = len(arg.init_s)
    #
    pdb_s = []
    box_s = []
    rsr_s = []
    try:
        for i in xrange(n_init):
            pdb = PDBFile(arg.init_s[i])
            pdb_s.append(pdb)
            box_s.append(np.loadtxt(arg.boxsize[i]) / 10.0)
            #
            if len(arg.rsr_fn_s) == 0:
                continue
            ref, rsr_info = read_custom_restraint(arg.rsr_fn_s[i])
            if ref is None: ref = pdb
            rsr_s.append(construct_custom_restraint(ref, rsr_info))
    except IndexError:
        sys.exit("ERROR: Number of initial PDB files, boxsize files, and restraint files must be identical.\n")
    #
    mdtraj_topology = mdtraj.load(arg.init_s[0]).topology
    if len(arg.dcdout) == 0:
        dcdIndex = mdtraj_topology.select("protein")
    else:
        dcdIndex = mdtraj_topology.select("protein or %s"%(' or '.join(['resname %s'%out for out in arg.dcdout])))
    #
    psf = CharmmPsfFile(arg.psf_fn)
    psf.setBox(*box_s[0])
    topology = psf.topology
    #
    ff = CharmmParameterSet(*arg.toppar)
    system = psf.createSystem(ff, nonbondedMethod=PME, \
                              switchDistance=0.8*nanometers,\
                              nonbondedCutoff=1.0*nanometers,\
                              constraints=HBonds)
    force_index = []
    if len(rsr_s) > 0:
        for restraint, info in rsr_s[0]:
            force_index.append(system.addForce(restraint))

    integrator = LangevinIntegrator(arg.temp*kelvin, \
                                    arg.langevin_friction_coefficient/picosecond,\
                                    arg.time[1]*picosecond)
    #
    runner = Restrained_WE_Runner(topology, system, integrator)
    #
    distance = calpha_rmsd
    #
    sampler = REVO_Sampler(system, topology, integrator, distance, \
                           p_max=arg.p_max, p_min=arg.p_min, d_merge=arg.d_merge)
    sampler.define_mdtraj_topology(mdtraj_topology)
    #
    if arg.restart is None:
        walker_s = []
        for i in xrange(arg.n_walker):
            k = i%n_init
            walker = Restrained_WE_Walker.initialize_with_runner(runner, pdb_s[k].positions, box=box_s[k])
            if len(rsr_s) > 0:
                walker.define_restraint([(force_index[j], X[1], X[0]) for j,X in enumerate(rsr_s[k])])
            walker_s.append(walker)
        #
        sampler.initialize_walker(walker_s)
    else:
        with open(arg.restart, 'rb') as fp:
            sampler.loadCheckpoint(fp)
    #
    walker_reporter_fn = '%s.walker'%arg.output_prefix
    log_fn = '%s.log'%arg.output_prefix
    dcd_fn = '%s.dcd'%arg.output_prefix
    pdb_fn = '%s.pdb'%arg.output_prefix
    pkl_fn = '%s.pkl'%arg.output_prefix
    #
    totalSteps = arg.n_cycle[0] * arg.n_cycle[1]
    sampler.set_walker_reporter(walker_reporter_fn)
    sampler.reporters.append(StateDataReporter(log_fn, 1, step=True, \
        time=True, kineticEnergy=True, potentialEnergy=True, temperature=True, progress=True, \
        remainingTime=True, speed=True, totalSteps=totalSteps, separator='\t'))
    sampler.reporters.append(mdtraj.reporters.DCDReporter(dcd_fn, 1, atomSubset=dcdIndex))
    sampler.reporters.append(PDBReporter(pdb_fn, totalSteps))
    #
    for _ in xrange(arg.n_cycle[0]):
        run_sampler(sampler, runner, arg.n_cycle[1], arg.time[0])
        for walker in sampler.walker_s:
            walker.update_restraint(mdtraj_topology)
    #
    with open(pkl_fn, 'wb') as fout:
        sampler.createCheckpoint(fout)
    #
    t_final = time.time()
    t_spend = t_final - t_init
    #
    with open(log_fn, 'at') as fout:
        fout.write("ELAPSED TIME:   %10.2f SECONDS\n"%t_spend)

def main():
    arg = argparse.ArgumentParser(prog='WE_sampler')
    arg.add_argument(dest='output_prefix')
    arg.add_argument('--psf', dest='psf_fn', required=True)
    arg.add_argument('--toppar', dest='toppar', nargs='*', required=True)
    #
    arg.add_argument('--init', dest='init_s', nargs='*', required=True)
    arg.add_argument('--boxsize', dest='boxsize', nargs='*', required=True)
    arg.add_argument('--rsr', dest='rsr_fn_s', nargs='*', default=[])
    arg.add_argument('--restart', dest='restart', default=None)
    arg.add_argument('--dcdout', dest='dcdout', nargs='*', default=[])
    #
    arg.add_argument('--n_walker', dest='n_walker', default=10, type=int)
    arg.add_argument('--n_cycle', dest='n_cycle', nargs=2, default=[0, 0], type=int)
    arg.add_argument('--time', dest='time', nargs=2, default=[50.0, 0.002], type=float)
    arg.add_argument('--p_max', dest='p_max', default=None, type=float)
    arg.add_argument('--p_min', dest='p_min', default=1e-12, type=float)
    arg.add_argument('--d_merge', dest='d_merge', default=2.5, type=float)
    #
    arg.add_argument('--temp', dest='temp', default=360.0, type=float)
    arg.add_argument('--friction', dest='langevin_friction_coefficient', default=0.01, type=float)
    #
    if len(sys.argv) == 1:
        arg.print_help()
        return
    arg = arg.parse_args()
    arg.time[0] = int(arg.time[0]/arg.time[1])
    if arg.p_max is None:
        arg.p_max = 1.0 / float(arg.n_walker) * 2.0
    #
    run(arg)

if __name__ == '__main__':
    main()
