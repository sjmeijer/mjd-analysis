import numpy as np
import os
import pickle

class FitConfiguration(object):
    def __init__(self,
        #data files
        wf_file_name="", field_file_name="",  conf_file="",
        wf_idxs=None,
        #save path
        directory = "",
        #params for setting up & aligning waveforms
        max_sample_idx = 200,
        numSamples = 400,
        fallPercentage = 0.97,
        alignType = "max",
        loadSavedConfig=False,
        avg_imp_guess = None,
        imp_grad_guess = None,
        beta_lims = [0, 1]
    ):

        self.wf_file_name=wf_file_name
        self.field_file_name=field_file_name
        self.siggen_conf_file=conf_file

        self.directory = directory

        self.wf_idxs = wf_idxs
        self.max_sample_idx = max_sample_idx
        self.fallPercentage = fallPercentage
        self.doMaxInterp = True

        #velocity model reference point field
        self.E_a = 500
        self.E_lo = 250
        self.E_hi = 1000

        if not (alignType == "max" or alignType == "timepoint"):
            print ("alignType must be 'max' or 'timepoint', not {0}".format(alignType))
            exit()
        self.alignType = alignType
        self.align_percent = 0.5
        self.numSamples = numSamples

        #limits & priors for the actual fit
        self.avg_imp_guess = avg_imp_guess
        self.imp_grad_guess = imp_grad_guess
        self.traprc_min = 150
        self.beta_lims = beta_lims

        if loadSavedConfig:
            self.load_config(directory)

    def save_config(self):
        saved_file=os.path.join(self.directory, "fit_params.npy")
        pickle.dump(self.__dict__.copy(),open(saved_file, 'wb'))

    def load_config(self,directory):
        saved_file=os.path.join(directory, "fit_params.npy")
        if not os.path.isfile(saved_file):
            print ("Saved configuration file {0} does not exist".format(saved_file))
            exit()

        self.__dict__.update(pickle.load(open(saved_file, 'rb')))
