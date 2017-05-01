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

    directory = "16wf_nofast"
    wf_file = "dat/P42661A_64_may1_nofast.npz"
    field_file = "dat/P42661A_apr27_21by21.npz"
    conf_file= "conf/P42661A_bull.conf"

    num_wfs = 16
    wf_idxs = list(range(0,64, int(64/num_wfs)))

    # #replace wf 0 with wf 1, because 0 looks bad maybe
    # wf_idxs[0] = 1

    conf = FitConfiguration(
        wf_file, field_file, conf_file, wf_idxs,
        directory = directory,
        alignType="timepoint",
        max_sample_idx = 100,
        numSamples = 250
    )

    required_particles = (num_wfs+1) * (num_particles) + 1
    if size != required_particles:
        sys.stdout.write("Your MPI global comm size is {0}.  Require {1} for {2} wfs and {3} particles\n".format(size, required_particles, num_wfs, num_particles))
        exit(0)

    fm = MPIFitManager(conf, comm = particle_comm, debug=False, doParallelParticles=True, )

    if rank == 0:
        conf.save_config()
    if rank ==0 or part_rank == 0:
        fm.fit_particle(manager_comm, numLevels=5000, numPerSave=1000, directory = directory, numParticles=num_particles)
        fm.close()
    else:
        fm.wait_and_process()


if __name__=="__main__":
    main()
