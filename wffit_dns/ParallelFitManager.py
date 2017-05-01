#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, shutil
import numpy as np
import dnest4

from Model import Model

from multiprocessing import Pool, cpu_count

class ParallelFitManager():
    '''Does the fit using one machine's parllel cores'''

    def __init__(self, fit_configuration, num_threads=None):

        self.model = Model( fit_configuration, fit_manager=self)
        self.num_waveforms = self.model.num_waveforms
        self.num_det_params = self.model.num_det_params

        if num_threads is None: num_threads = cpu_count()

        if num_threads > self.model.num_waveforms: num_threads = self.model.num_waveforms

        self.pool = Pool(num_threads)

    def calc_likelihood(self, params):
        num_det_params = self.num_det_params


        wfs_param_arr = params[num_det_params:].reshape((6, self.num_waveforms))

        wf_params = np.empty(num_det_params+6)
        wf_params[:num_det_params] = params[:num_det_params]

        args = []
        #parallelized calculation
        for wf_idx in range(self.num_waveforms):
            my_wf_params = np.copy(wf_params)
            wf_params[num_det_params:] = wfs_param_arr[:,wf_idx]
            args.append( [my_wf_params, wf_idx)

        results = pool.map(self.WaveformLogLikeStar, args)
        results = pool.map(WaveformLogLikeStar, args)

        lnlike = 0
        for result in (results):
            lnlike += result

        return np.sum(wf_likes)

    def WaveformLogLikeStar(self, a_b):
      return self.model.calc_wf_likelihood(*a_b)
