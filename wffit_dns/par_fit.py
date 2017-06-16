#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demonstration of DNest4 in Python using the "StraightLine" example
"""
import sys, os, shutil
import numpy as np

import dnest4

from ParallelFitManager import ParallelFitManager
from FitConfiguration import FitConfiguration

def main():
    directory = "16wf_P42538A"
    wf_file = "dat/P42538A_64_slow.npz"
    field_file = "dat/P42538A_bull_fields.npz"
    conf_file= "conf/P42538A_bull.conf"
    num_wfs = 4
    wf_idxs = list(range(0,64,16))

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

    fm = ParallelFitManager(conf, )

    conf.save_config()
    fm.fit(numLevels=1000, directory = directory,new_level_interval=10000)


if __name__=="__main__":
    main()
