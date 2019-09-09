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
from test import Restrained_WE_Runner, Restrained_WE_Walker

FF = glob.glob("%s/ff/c36m/*"%(os.environ['work']))

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
    system = psf.createSystem(ff, nonbondedMethod=PME, \
                              switchDistance=0.8*nanometers,\
                              nonbondedCutoff=1.0*nanometers,\
                              constraints=HBonds)

    restraint_info = []
    for restraint, info in restraint_s:
        force_index = system.addForce(restraint)
        restraint_info.append((force_index, info, restraint))

    topology = psf.topology
    integrator = LangevinIntegrator(300*kelvin, 0.01/picosecond, 0.002*picosecond)
    #
    if MPI_RANK == MPI_KING:
        walker_s = [Restrained_WE_Walker(pdb.positions) for _ in range(20)]
        for walker in walker_s:
            walker.define_restraint(restraint_info)
        #
        sampler = REVO_Sampler(system, topology, integrator, calpha_rmsd)
        sampler.initialize_walker(walker_s)
        #
        TOP = mdtraj.load(pdb_fn).topology
        sampler.define_mdtraj_topology(TOP)
        #
        sampler.reporters.append(StateDataReporter('test3.log', 1, step=True, \
            time=True, kineticEnergy=True, potentialEnergy=True, temperature=True, progress=True, \
            remainingTime=True, speed=True, totalSteps=100, separator='\t'))
        sampler.reporters.append(DCDReporter("test3.dcd", 1))
        sampler.reporters.append(PDBReporter("test3.pdb", 100))
        #
        runner = None
    else:
        sampler = None
        runner = Restrained_WE_Runner(topology, system, integrator)

    MPI_COMM.barrier()

    #run_sampler(sampler, runner, 5, 5000)
    for _ in range(10):
        run_sampler(sampler, runner, 20, 25000)
        for walker in sampler.walker_s:
            walker.update_restraint(TOP)
    #
    if MPI_RANK == MPI_KING:
        with open("test3.pkl", 'wb') as fout:
            sampler.createCheckpoint(fout)

    MPI_COMM.barrier()

if __name__ == '__main__':
    main()
