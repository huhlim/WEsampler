#!/usr/bin/python

import os
import sys
import glob
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import mdtraj

from libcustom import read_custom_restraint, construct_custom_restraint

from simtk.unit import *
from simtk.openmm import *
from simtk.openmm.app import *

from libmpi import *
from libwe import REVO_Sampler, WE_Walker, WE_Runner, run_sampler

FF = glob.glob("%s/ff/c36m/*"%(os.environ['work']))

class Restrained_WE_Walker(WE_Walker):
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
    def walker_to_simulation(self, walker):
        super(Restrained_WE_Runner, self).walker_to_simulation(walker)
        walker.feed_restraint(self.simulation)

def calpha_rmsd(x,y, top):
    calphaIndex = top.select("name CA")
    return mdtraj.rmsd(x, y, atom_indices=calphaIndex)[0] * 10.0

def main():
    pdb_fn = 'TR894.equil.pdb'
    psf_fn = 'TR894.psf'
    boxsize = np.loadtxt('boxsize') / 10.0
    rsr_fn = 'TR894.rsr'
    #
    ff = CharmmParameterSet(*FF)
    psf = CharmmPsfFile(psf_fn)
    psf.setBox(*boxsize)
    #
    pdb = PDBFile(pdb_fn)
    #
    custom_restraints = read_custom_restraint(rsr_fn)
    restraint_s = construct_custom_restraint(pdb, custom_restraints)
    #
    TOP = mdtraj.load(pdb_fn).topology
    walker_s = [Restrained_WE_Walker(pdb.positions) for _ in range(20)]
    #
    system = psf.createSystem(ff, nonbondedMethod=PME, \
                              switchDistance=0.8*nanometers,\
                              nonbondedCutoff=1.0*nanometers,\
                              constraints=HBonds)

    restraint_info = []
    for restraint, info in restraint_s:
        force_index = system.addForce(restraint)
        restraint_info.append((force_index, info, restraint))
    for walker in walker_s:
        walker.define_restraint(restraint_info)

    topology = psf.topology
    integrator = LangevinIntegrator(300*kelvin, 0.01/picosecond, 0.002*picosecond)
    #
    distance = calpha_rmsd
    #
    sampler = REVO_Sampler(system, topology, integrator, distance, p_max=0.1, d_merge=2.5)
    sampler.initialize_walker(walker_s)
    sampler.define_mdtraj_topology(TOP)
    sampler.reporters.append(StateDataReporter('test2.log', 1, step=True, \
        time=True, kineticEnergy=True, potentialEnergy=True, temperature=True, progress=True, \
        remainingTime=True, speed=True, totalSteps=200, separator='\t'))
    sampler.reporters.append(DCDReporter("test2.dcd", 1))
    sampler.reporters.append(PDBReporter("test2.pdb", 200))
    #
    runner = Restrained_WE_Runner(topology, system, integrator)

    for _ in range(10):
        run_sampler(sampler, runner, 20, 25000)
        for walker in sampler.walker_s:
            walker.update_restraint(TOP)
    #
    with open("test2.pkl", 'wb') as fout:
        sampler.createCheckpoint(fout)

if __name__ == '__main__':
    main()
