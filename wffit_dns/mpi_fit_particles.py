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

num_particles = 5
#
comm = MPI.COMM_WORLD   # get MPI communicator object
size = comm.size        # total number of processes
rank = comm.rank        # rank of this process
status = MPI.Status()   # get MPI status object

#
proc_per_part = (size-1) / np.float(num_particles)

if not proc_per_part.is_integer():
    sys.stdout.write("size is %d, processes per particle is %f. must be integer\n" % (size, proc_per_part))
    exit(0)
else:
    proc_per_part =np.int(proc_per_part)

full_group = comm.Get_group()
particle_mgr = np.int(np.floor( (rank-1) / proc_per_part) * proc_per_part + 1)
part_rank = (rank-1) % proc_per_part

managers = np.arange(1,size, proc_per_part, dtype="int")
managers = np.insert(managers, 0,0)
manager_group = MPI.Group.Incl(full_group,managers.tolist())
manager_comm = comm.Create(manager_group)

if rank != 0:
    group_idxs = np.arange( np.int(particle_mgr), np.int(particle_mgr+proc_per_part), dtype="int")
else:
    # print "manager group" + str(managers)
    group_idxs = np.arange(2,np.int(1+proc_per_part), dtype="int")
    group_idxs = np.insert(group_idxs, 0,0)

particle_group = MPI.Group.Incl(full_group,group_idxs.tolist())
particle_comm = comm.Create(particle_group)
#sys.stdout.write("rank %d group_idxs: %s\n" % (rank, str(group_idxs)))


def main():

    # directory = "16wf_final4"
    # wf_file = "dat/P42661A_64_slow.npz"
    # # wf_file = "dat/P42661A_64_may2_nofast.npz"
    # # field_file = "dat/P42661A_fine_fields.npz"
    # # conf_file= "conf/P42661A_fine.conf"
    # # field_file = "dat/P42661A_bull_pcfields_wideimp.npz"
    #
    # # field_file = "dat/P42661A_bull_may26_fields.npz"
    # field_file = "dat/P42661A_bull_may24_fields.npz"
    # conf_file= "conf/P42661A_bull.conf"
    #
    # num_wfs = 16
    # # all_wfs = list(range(64))
    # # good_wfs = np.delete(all_wfs, [ 24, 25,  56, 54, 46, 38, 26, 27, 44], axis=0)
    # # wf_idxs = good_wfs[::np.int(64/num_wfs)]
    # #
    # # for val in [23,45]:
    # #     insert_idx = np.searchsorted(wf_idxs, val)
    # #     wf_idxs = np.insert(wf_idxs, insert_idx, val)
    #
    # #before was 1,5
    # wf_idxs = [ 0,  9,  8, 12, 16, 20, 23, 25, 31, 35, 40, 42, 49, 52, 57, 62]

    directory = "16wf_P42538A"
    wf_file = "dat/P42538A_64_slow.npz"
    field_file = "dat/P42538A_bull_fields.npz"
    conf_file= "conf/P42538A_bull.conf"
    num_wfs = 16
    wf_idxs = list(range(0,64,4))

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

    required_particles = (num_wfs+1) * (num_particles) + 1
    if size != required_particles:
        sys.stdout.write("Your MPI global comm size is {0}.  Require {1} for {2} wfs and {3} particles\n".format(size, required_particles, num_wfs, num_particles))
        exit(0)

    fm = MPIFitManager(conf, comm = particle_comm, debug=False, doParallelParticles=True, )

    if rank == 0:
        conf.save_config()
    if rank ==0 or part_rank == 0:
        fm.fit_particle(manager_comm, numLevels=5000, numPerSave=1000, directory = directory, numParticles=num_particles, new_level_interval=10000)
        fm.close()
    else:
        fm.wait_and_process()


if __name__=="__main__":
    main()
