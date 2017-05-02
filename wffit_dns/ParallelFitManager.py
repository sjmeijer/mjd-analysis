#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, shutil
import numpy as np
import dnest4

from Model import Model

from multiprocessing import Pool, cpu_count

def init_parallelization(conf):
    global model
    model = Model( conf,)
def WaveformLogLikeStar(a_b):
  return model.calc_wf_likelihood(*a_b)

class ParallelFitManager():
    '''Does the fit using one machine's parallel cores'''

    def __init__(self, fit_configuration, num_threads=None):

        self.model = Model( fit_configuration, fit_manager=self)
        self.num_waveforms = self.model.num_waveforms
        self.num_det_params = self.model.num_det_params

        if num_threads is None: num_threads = cpu_count()

        if num_threads > self.model.num_waveforms: num_threads = self.model.num_waveforms

        self.pool = Pool(num_threads, initializer=init_parallelization, initargs=(fit_configuration,))

    def calc_likelihood(self, params):
        num_det_params = self.num_det_params

        wfs_param_arr = params[num_det_params:].reshape((6, self.num_waveforms))
        wf_params = np.zeros((num_det_params+6,self.num_waveforms))

        args = []
        #parallelized calculation
        for wf_idx in range(self.num_waveforms):
            wf_params[:num_det_params,wf_idx] = params[:num_det_params]
            wf_params[num_det_params:,wf_idx] = wfs_param_arr[:,wf_idx]
            args.append( [wf_params[:,wf_idx], wf_idx])
            # print ("shipping {0}: {1}".format(wf_idx, wf_params[num_det_params:, wf_idx]))
        results = self.pool.map(WaveformLogLikeStar, args)
        # exit()
        lnlike = 0
        for result in (results):
            lnlike += result
            # print (result)
        return lnlike

    def fit(self, numLevels, directory="",numPerSave=1000,numParticles=5 ):

      sampler = dnest4.DNest4Sampler(self.model,
                                     backend=dnest4.backends.CSVBackend(basedir ="./" + directory,
                                                                        sep=" "))

      # Set up the sampler. The first argument is max_num_levels
      gen = sampler.sample(max_num_levels=numLevels, num_steps=200000, new_level_interval=10000,
                            num_per_step=numPerSave, thread_steps=100,
                            num_particles=numParticles, lam=10, beta=100, seed=1234)

      # Do the sampling (one iteration here = one particle save)
      for i, sample in enumerate(gen):
          print("# Saved {k} particles.".format(k=(i+1)))
