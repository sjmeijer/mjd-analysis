#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demonstration of DNest4 in Python using the "StraightLine" example
"""
import sys, os, shutil
import numpy as np

import dnest4

from MPIFitManager import MPIFitManager
from FitConfiguration import FitConfiguration
from mpi4py import MPI

comm = MPI.COMM_WORLD   # get MPI communicator object
rank = comm.Get_rank()

def main():
    directory = "test"
    wf_file = "dat/P42661A_64_may1.npz"
    field_file = "dat/P42661A_apr27_21by21.npz"
    conf_file= "conf/P42661A_bull.conf"

    wf_num = 4
    wf_idxs = range(0,64, np.int(64/4))

    conf = FitConfiguration(
        wf_file, field_file, conf_file, wf_idxs,
        directory = directory,
        alignType="timepoint",
        max_sample_idx = 100,
        numSamples = 250
    )

    fm = MPIFitManager(conf, comm = comm, debug=False)

    if rank == 0:
        conf.save_config()
        fm.fit(numLevels=1000, directory = directory)
        fm.close()
    else:
        fm.wait_and_process()


if __name__=="__main__":
    main()
