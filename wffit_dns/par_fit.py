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
    directory = "8wf_noalias"
    wf_file = "dat/P42661A_64_may2_nofast.npz"
    field_file = "dat/P42661A_may1_21by21.npz"
    conf_file= "conf/P42661A_bull.conf"

    wf_num = 8
    offset = 1
    wf_idxs = range(offset,64+offset, np.int(64/wf_num))

    # wf_idxs = [0, 8*7]

    conf = FitConfiguration(
        wf_file, field_file, conf_file, wf_idxs,
        directory = directory,
        alignType="timepoint",
        max_sample_idx = 125,
        numSamples = 250
    )

    fm = ParallelFitManager(conf, )

    conf.save_config()
    fm.fit(numLevels=1000, directory = directory)


if __name__=="__main__":
    main()
