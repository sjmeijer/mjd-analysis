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
    directory = "8wf_P42574B"
    # wf_file = "dat/P42538A_64_slow.npz"
    wf_file = "dat/P42574B_8_slow.npz"
    field_file = "dat/P42574B_bull_fields.npz"
    conf_file= "conf/P42574B.conf"
    # wf_file = "dat/P42574B_64_slow.npz"
    # field_file = "dat/P42538A_bull_fields.npz"
    # conf_file= "conf/P42538A_bull.conf"
    num_wfs = 8
    wf_idxs = list(range(0,8,4))
    
    conf = FitConfiguration(
        wf_file, field_file, conf_file, wf_idxs,
        directory = directory,
        alignType="timepoint",
        max_sample_idx = 125,
        numSamples = 250,
        imp_grad_guess= 0.1,
        avg_imp_guess= -0.408716,
        interpType = "linear",
        smooth_type = "gen_gaus"
    )

    fm = MPIFitManager(conf, comm = comm, debug=False)

    if rank == 0:
        conf.save_config()
        fm.fit(numLevels=1000, directory = directory, numParticles=5)
        fm.close()
    else:
        fm.wait_and_process()


if __name__=="__main__":
    main()
