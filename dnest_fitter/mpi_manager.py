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

from mpi4py import MPI


verbose = 0
doMaxInterp = True

class MPIFitManager():
    def __init__(self, wf_file_name, wf_idxs, comm=None, doParallelParticles = False, debug=False):

        if comm is None:
            self.comm = MPI.COMM_WORLD
        else:
            self.comm = comm

        self.doParallelParticles = doParallelParticles

        self.rank = self.comm.Get_rank()
        self.num_workers = self.comm.Get_size() - 1

        self.max_sample_idx = 200
        self.fallPercentage = 0.97

        self.wf_idxs = wf_idxs

        # if size != (len(self.wf_idxs) + 1) * num_particles +1:
        #     print "not the right number of mpi processes!"
        #     exit(0)

        self.wf_file_name = wf_file_name
        self.field_file_name = "P42574A_mar28_21by21.npz"
        self.det_name = "conf/P42574A_bull.conf"

        self.tags = self.enum('CALC_LIKE', 'CALC_WF', 'EXIT')

        self.setup_waveforms(self.wf_file_name)
        self.setup_detector(self.field_file_name)


        # if self.num_workers != self.num_waveforms:
        #     print "Should have same number of workers (%d) as waveforms (%d)!!" % (self.num_workers, self.num_waveforms)
        #     exit(0)

        self.E_a = 500

        self.charge_wf_task_len = 11
        self.extra_process_params = 13

        self.num_workers = self.comm.size - 1

        self.numCalls = 0
        self.LastMem = memory_usage_psutil()

        self.debug = debug
        self.debug_mem_file = "memory_info.txt"

    def is_master(self):
        """
        Is the current process the master?
        """
        return self.rank == 0

    def save_fit_params(self, directory):
        np.savez(directory+"fit_params.npz",
            max_sample_idx = self.max_sample_idx,
            fallPercentage=self.fallPercentage,
            wf_idxs=self.wf_idxs,
            wf_file_name=self.wf_file_name,
            field_file_name=self.field_file_name,
            doMaxInterp=doMaxInterp,
            det_name = self.det_name
            )

    def setup_waveforms(self, wfFileName):

        if os.path.isfile(wfFileName):
            data = np.load(wfFileName, encoding="latin1")
            wfs = data['wfs']

            wfs = wfs[self.wf_idxs]

            self.wfs = wfs
            self.num_waveforms = wfs.size
        else:
          print("Saved waveform file %s not available" % wfFileName)
          exit(0)

        wfLengths = np.empty(wfs.size)
        wfMaxes = np.empty(wfs.size)
        baselineLengths = np.empty(wfs.size)

        for (wf_idx,wf) in enumerate(wfs):
          wf.WindowWaveformAroundMax(fallPercentage=self.fallPercentage, rmsMult=2, earlySamples=self.max_sample_idx)
          baselineLengths[wf_idx] = wf.t0Guess

          if MPI.COMM_WORLD.Get_rank() == 0:
              print( "wf %d length %d (entry %d from run %d)" % (wf_idx, wf.wfLength, wf.entry_number, wf.runNumber))
          wfLengths[wf_idx] = wf.wfLength
          wfMaxes[wf_idx] = np.argmax(wf.windowedWf)

        self.baseline_origin_idx = np.amin(baselineLengths) - 30
        if self.baseline_origin_idx < 0:
            print( "not enough baseline!!")
            exit(0)

        self.siggen_wf_length = np.int(  (self.max_sample_idx - np.amin(baselineLengths) + 10)*10  )
        self.output_wf_length = np.int( np.amax(wfLengths) + 1 )

        if MPI.COMM_WORLD.Get_rank() == 0:
            print( "siggen_wf_length will be %d, output wf length will be %d" % (self.siggen_wf_length, self.output_wf_length))


    def setup_detector(self, fieldFileName):
        timeStepSize = 1 #ns
        det =  Detector(self.det_name, timeStep=timeStepSize, numSteps=self.siggen_wf_length, maxWfOutputLength =self.output_wf_length, t0_padding=100 )

        det.LoadFieldsGrad(fieldFileName)

        self.detector = det

    def calc_likelihood(self, params_in):
        params = np.copy(params_in)
        num_waveforms = self.num_waveforms
        tags = self.tags

        if self.debug:
            self.numCalls +=1
            if self.numCalls % 1000 == 0:
                meminfo = "Particle {0} (call {1}) memory: {2}\n".format(MPI.COMM_WORLD.Get_rank(), self.numCalls , memory_usage_psutil())
                with open(self.debug_mem_file, "a") as f:
                    f.write(meminfo)

        (tf_first_idx, velo_first_idx, trap_idx, grad_idx, num_det_params) = self.indices

        phi, omega, d, rc1, rc2, rcfrac, aliasrc = params[tf_first_idx:tf_first_idx+7]
        h_111_va, h_111_vmax, h_100_multa, h_100_multmax, h_100_beta, h_111_beta, = params[velo_first_idx:velo_first_idx+6]
        charge_trapping = params[trap_idx]
        grad = params[grad_idx]
        avg_imp = params[grad_idx+1]

        rad_arr, phi_arr, theta_arr, scale_arr, t0_arr, smooth_arr, = params[num_det_params:].reshape((6, num_waveforms))

        if self.doParallelParticles and MPI.COMM_WORLD.Get_rank() == 0:

            ln_like = 0
            for wf_idx in range(self.num_waveforms):
                wf_params = rad_arr[wf_idx], phi_arr[wf_idx], theta_arr[wf_idx], \
                              scale_arr[wf_idx], t0_arr[wf_idx], smooth_arr[wf_idx],\
                              phi, omega, d, rc1, rc2, rcfrac,aliasrc,\
                              h_111_va, h_111_vmax, h_100_multa, h_100_multmax, h_100_beta, h_111_beta,\
                              grad, avg_imp, charge_trapping
                ln_like += self.calc_wf_likelihood(wf_params, wf_idx)
            return ln_like

        # requests = []
        for wf_idx in range(self.num_waveforms):
            worker = np.int(wf_idx + 1)
            wf_params = rad_arr[wf_idx], phi_arr[wf_idx], theta_arr[wf_idx], \
                          scale_arr[wf_idx], t0_arr[wf_idx], smooth_arr[wf_idx],\
                          phi, omega, d, rc1, rc2, rcfrac,aliasrc,\
                          h_111_va, h_111_vmax, h_100_multa, h_100_multmax, h_100_beta, h_111_beta,\
                          grad, avg_imp, charge_trapping

            self.comm.send(wf_params, dest=worker, tag=self.tags.CALC_LIKE)
            # requests.append(r)

        # MPI.Request.waitall(requests)

        wf_likes = np.empty(self.num_waveforms)
        for i in range(self.num_waveforms):
                worker = i + 1
                wf_likes[i] = self.comm.recv(source=worker, tag=MPI.ANY_TAG)

        return np.sum(wf_likes)

    def make_waveform(self, data_len, wf_params):
        bl_origin_idx = self.baseline_origin_idx

        rad, phi, theta, scale, maxt, smooth, \
        tf_phi, tf_omega, d, rc1, rc2, rcfrac, aliasrc,\
        h_111_va, h_111_vmax, h_100_multa, h_100_multmax, h_100_beta, h_111_beta,\
        grad, avg_imp, charge_trapping = wf_params

        r = rad * np.cos(theta)
        z = rad * np.sin(theta)

        c = -d * np.cos(tf_omega)
        b_ov_a = c - np.tan(tf_phi) * np.sqrt(d**2-c**2)
        a = 1./(1+b_ov_a)
        tf_b = a * b_ov_a

        h_100_va = h_100_multa * h_111_va
        h_100_vmax = h_100_multmax * h_111_vmax

        h_100_mu0, h_100_beta, h_100_e0 = self.get_velo_params(h_100_va, h_100_vmax, h_100_beta)
        h_111_mu0, h_111_beta, h_111_e0 = self.get_velo_params(h_111_va, h_111_vmax, h_111_beta)

        if scale < 0:
            return None
        if smooth < 0:
            return None
        if not self.detector.IsInDetector(r, phi, z):
            return None

        self.detector.SetTransferFunction(tf_b, c, d, rc1, rc2, rcfrac, )
        self.detector.siggenInst.set_hole_params(h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0)
        self.detector.trapping_rc = charge_trapping

        self.detector.SetAntialiasingRC(aliasrc)

        self.detector.SetGrads(grad, avg_imp)

        model = self.detector.MakeSimWaveform(r, phi, z, scale, maxt, data_len, h_smoothing=smooth, alignPoint="max", doMaxInterp=doMaxInterp)

        if model is None or np.any(np.isnan(model)):
            return None

        # start_idx = -bl_origin_idx
        # end_idx = data_len - bl_origin_idx - 1
        # baseline_trend = np.linspace(m*start_idx+b, m*end_idx+b, data_len)
        # model += baseline_trend

        return model

    def wait_and_process(self):
        detector = self.detector
        tags = self.tags
        status = MPI.Status()   # get MPI status object

        if self.is_master():
            raise RuntimeError("Master node told to await jobs.")

        while True:
            # Blocking receive to wait for instructions.
            task = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

            if self.debug:
                self.numCalls +=1
                if self.numCalls % 1000 == 0:
                    meminfo = "Particle {0} (call {1}) memory: {2}\n".format(MPI.COMM_WORLD.Get_rank(), self.numCalls , memory_usage_psutil())
                    with open(self.debug_mem_file, "a") as f:
                        f.write(meminfo)

            if status.tag == self.tags.CALC_LIKE:
                # if self.debug:
                #     print( "rank %d (local rank %d) calcing like %d" % (MPI.COMM_WORLD.Get_rank(), self.rank, self.rank - 1) )

                wf_idx = self.rank - 1
                ln_like = self.calc_wf_likelihood(task, wf_idx)
                self.comm.send(ln_like, dest=0, tag=status.tag)

            if status.tag == self.tags.CALC_WF:
                data_len = self.output_wf_length
                model = self.make_waveform(data_len, task)

                self.comm.send(model, dest=0, tag=status.tag)

            elif status.tag == self.tags.EXIT:
                break

            del task

    def calc_wf_likelihood(self, params, wf_idx ):
        wf = self.wfs[wf_idx]
        data = wf.windowedWf
        model_err = wf.baselineRMS
        data_len = len(data)
        model = self.make_waveform(data_len, params)

        if model is None:
            ln_like = -np.inf
        else:
            inv_sigma2 = 1.0/(model_err**2)
            ln_like = -0.5*(np.sum((data-model)**2*inv_sigma2 - np.log(inv_sigma2)))

        return ln_like

    def enum(self, *sequential, **named):
        """Handy way to fake an enumerated type in Python
        http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
        """
        enums = dict(zip(sequential, range(len(sequential))), **named)
        return type('Enum', (), enums)

    def set_indices(self, indices):
        self.indices = indices

    def fit(self, model, numLevels, directory=""):

      sampler = dnest4.DNest4Sampler(model,
                                     backend=dnest4.backends.CSVBackend(basedir ="./" + directory,
                                                                        sep=" "))

      # Set up the sampler. The first argument is max_num_levels
      gen = sampler.sample(max_num_levels=numLevels, num_steps=200000, new_level_interval=10000,
                            num_per_step=1000, thread_steps=100,
                            num_particles=5, lam=10, beta=100, seed=1234)

      # Do the sampling (one iteration here = one particle save)
      for i, sample in enumerate(gen):
          print("# Saved {k} particles.".format(k=(i+1)))

      # Run the postprocessing
      # dnest4.postprocess()

    def fit_particle(self, manager_comm, model, numLevels, directory="", numPerSave=1000):

      mpi_sampler = dnest4.MPISampler(comm=manager_comm, debug=False)

      if manager_comm.rank == 0:
          # Set up the sampler. The first argument is max_num_levels
          sampler = dnest4.DNest4Sampler(model, backend=dnest4.backends.CSVBackend(basedir ="./" + directory,
                                                                          sep=" "), MPISampler=mpi_sampler)

          gen = sampler.sample(max_num_levels=numLevels, num_steps=200000, new_level_interval=10000,
                                num_per_step=1000, thread_steps=100,
                                lam=10, beta=100, seed=1234)

          # Do the sampling (one iteration here = one particle save)
          for i, sample in enumerate(gen):
            #   print("# Saved {k} particles.".format(k=(i+1)))

              if self.debug:
                  meminfo = "Particle {0} memory: {1}\n".format(MPI.COMM_WORLD.Get_rank(),  memory_usage_psutil())
                  with open(self.debug_mem_file, "a") as f:
                      f.write(meminfo)

      else:
          mpi_sampler.wait(model, max_num_levels=numLevels, num_steps=200000, new_level_interval=10000,
                                num_per_step=numPerSave, thread_steps=100,
                                lam=10, beta=100, seed=1234)
          return

    def close(self):
        if self.is_master():
            for i in range(self.num_workers):
                self.comm.send(None, dest=i + 1, tag=self.tags.EXIT)

    def get_velo_params(self, v_a, v_max, beta):
        E_a = self.E_a
        E_0 = np.power( (v_max*E_a/v_a)**beta - E_a**beta , 1./beta)
        mu_0 = v_max / E_0

        return (mu_0,  beta, E_0)

    def __exit__(self, *args):
        self.close()

def memory_usage_psutil():
    # return the memory usage in MB
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem
