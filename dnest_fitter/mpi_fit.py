#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demonstration of DNest4 in Python using the "StraightLine" example
"""
import sys, os, shutil
import numpy as np

from helpers import Waveform
from pysiggen import Detector

from dns_model import Model
import dnest4

from mpi_manager import MPIFitManager
from mpi4py import MPI

comm = MPI.COMM_WORLD   # get MPI communicator object
size = comm.size        # total number of processes
rank = comm.rank        # rank of this process


verbose = 0
doMaxInterp = True

def main():
    directory = "64wf_nopart/"

    wf_idxs = range(0,64)
    fm = MPIFitManager("P42574A_64_spread.npz", wf_idxs, comm = comm)

    if rank == 0:
        fm.save_fit_params(directory)
        model = Model(fm)
        fm.set_indices( model.get_indices() )

        fm.fit(model, numLevels=1000, directory = directory)

        fm.close()
    else:
        fm.wait_and_process()


if __name__=="__main__":
    main()
