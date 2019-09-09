#!/usr/bin/env python

import os
from mpi4py import MPI

MPI_COMM = MPI.COMM_WORLD
MPI_SIZE = MPI_COMM.Get_size()
MPI_RANK = MPI_COMM.Get_rank()
MPI_KING = 0
MPI_ANY_SOURCE = MPI.ANY_SOURCE

if 'CUDA_VISIBLE_DEVICES' in os.environ:
    GPU_s = os.environ['CUDA_VISIBLE_DEVICES'].split(",")
else:
    GPU_s = ['0']
n_GPU = len(GPU_s)
MPI_GPU_BINDING = [None] + [GPU_s[i%n_GPU] for i in xrange(MPI_SIZE-1)]
