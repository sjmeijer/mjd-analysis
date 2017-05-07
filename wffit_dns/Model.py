import numpy as np
import scipy.stats as stats
import dnest4
import os

from Waveform import Waveform
from pysiggen import Detector
import numpy.random as rng

tf_first_idx = 0
velo_first_idx = 6

grad_idx = velo_first_idx + 6
imp_avg_idx = grad_idx + 1

pcrad_idx, pclen_idx = imp_avg_idx+1, imp_avg_idx+2
trap_idx = imp_avg_idx + 3

phi_idx, omega_idx, d_idx = np.arange(3)+ tf_first_idx
rc1_idx, rc2_idx, rcfrac_idx = np.arange(3)+ tf_first_idx+3
# aliasrc_idx = tf_first_idx+6

class Model(object):
    """
    Specify the model in Python.
    """
    def __init__(self, fit_configuration, fit_manager=None):

        self.conf = fit_configuration
        self.fit_manager = fit_manager


        #Setup detector and waveforms
        self.setup_waveforms(doPrint=False)
        self.setup_detector()

        self.alignidx_guess = self.conf.max_sample_idx
        self.max_maxt = self.alignidx_guess + 5
        self.min_maxt = self.alignidx_guess - 5
        self.maxt_sigma = 1

        self.changed_wfs = np.zeros(self.num_waveforms)

        self.num_det_params = trap_idx+1

        #set up prior information (for gausian prior vars)
        priors = np.empty(self.num_det_params) #6 + 2)
        prior_vars =  np.empty_like(priors)

        priors[phi_idx], prior_vars[phi_idx] = -1.5, 0.2
        priors[omega_idx], prior_vars[omega_idx] = 0.11, 0.1
        priors[d_idx], prior_vars[d_idx] = 0.8, 0.05

        priors[rc1_idx], prior_vars[rc1_idx] = self.rc1_guess, 0.2
        priors[rc2_idx], prior_vars[rc2_idx] = self.rc2_guess, 0.3
        priors[rcfrac_idx], prior_vars[rcfrac_idx] = self.rcfrac_guess, 0.01

        #velo params from reggiani
        # h100_v500_meas = 7.156476E6
        # h111_v500_meas = 6.056016E6
        # h100_mu0E0_meas = 12.006273E6
        # h111_mu0E0_meas = 10.727000E6
        #
        # priors[velo_first_idx] = h100_v500_meas
        # priors[velo_first_idx+1] = h111_v500_meas
        # priors[velo_first_idx+2] = h100_mu0E0_meas
        # priors[velo_first_idx+3] = h111_mu0E0_meas

        h100_250= 5.50460266087E6
        h111_250= 4.83393529058E6
        h100_1000= 8.61282416937E6
        h111_1000= 7.17181419648E6

        # h100_100 = 3.40293878765E6
        # h111_100 = 3.2468411889E6
        # h100_3000 = 10.2630330467E6
        # h111_3000 = 8.56978532533E6

        priors[velo_first_idx] = h100_250
        priors[velo_first_idx+1] = h111_250
        priors[velo_first_idx+2] = h100_1000
        priors[velo_first_idx+3] = h111_1000

        for i in range(4):
            prior_vars[velo_first_idx+i] = 0.2*priors[velo_first_idx+i]

        # if self.conf.avg_imp_guess is None:
        #     priors[imp_avg_idx] = self.detector.measured_impurity
        # else:
        #     priors[imp_avg_idx] = self.conf.avg_imp_guess
        #
        # if self.conf.avg_imp_guess is None:
        #     priors[grad_idx] = self.detector.measured_imp_grad
        # else:
        #     priors[grad_idx] = self.conf.imp_grad_guess
        #
        # prior_vars[imp_avg_idx] = np.abs(0.2* priors[imp_avg_idx])
        # prior_vars[grad_idx] =  np.amax((0.2*priors[grad_idx], 0.1))


        self.priors = priors
        self.prior_vars = prior_vars

    def setup_detector(self):
        timeStepSize = 1 #ns
        det =  Detector(self.conf.siggen_conf_file, timeStep=timeStepSize, numSteps=self.siggen_wf_length, maxWfOutputLength =self.output_wf_length, t0_padding=100 )
        det.LoadFieldsGrad(self.conf.field_file_name)

        self.detector = det

    def setup_waveforms(self, doPrint=False):
        wfFileName = self.conf.wf_file_name

        if os.path.isfile(wfFileName):
            print("Loading wf file {0}".format(wfFileName))
            data = np.load(wfFileName, encoding="latin1")
            wfs = data['wfs']
            wfs = wfs[self.conf.wf_idxs]

            self.rc1_guess = data['rc1']
            self.rc2_guess =data['rc2']
            self.rcfrac_guess =data['rcfrac']

            self.wfs = wfs
            self.num_waveforms = wfs.size
        else:
          print("Saved waveform file %s not available" % wfFileName)
          exit(0)

        wfLengths = np.empty(wfs.size)
        wfMaxes = np.empty(wfs.size)

        baselineLengths = np.zeros(wfs.size)

        for (wf_idx,wf) in enumerate(wfs):
          if self.conf.alignType == "max":
              wf.WindowWaveformAroundMax(fallPercentage=self.conf.fallPercentage, rmsMult=2, earlySamples=self.conf.max_sample_idx)

          elif self.conf.alignType == "timepoint":
              wf.WindowWaveformAroundTimepoint(earlySamples=self.conf.max_sample_idx,  timePoint=self.conf.align_percent, numSamples=self.conf.numSamples, rmsMult=2)

          if doPrint:
              print( "wf %d length %d (entry %d from run %d)" % (wf_idx, wf.wfLength, wf.entry_number, wf.runNumber))
          wfLengths[wf_idx] = wf.wfLength
          wfMaxes[wf_idx] = np.argmax(wf.windowedWf)
          baselineLengths[wf_idx] = wf.t0Guess

        # import matplotlib.pyplot as plt
        # plt.figure()
        # for wf in wfs:
        #     plt.plot(wf.windowedWf)
        # plt.show()
        # plt.exit()

        # self.baseline_origin_idx = np.amin(baselineLengths) - 30
        # if self.baseline_origin_idx < 0:
        #     print( "not enough baseline!!")
        #     exit(0)

        self.siggen_wf_length = np.int(  (self.conf.max_sample_idx - np.amin(baselineLengths) + 10)*10  )
        self.output_wf_length = np.int( np.amax(wfLengths) + 1 )

        if doPrint:
            print( "siggen_wf_length will be %d, output wf length will be %d" % (self.siggen_wf_length, self.output_wf_length))


    def draw_position(self, wf_idx):
      r = rng.rand() * self.detector.detector_radius
      z = rng.rand() * self.detector.detector_length
      scale = np.amax(self.wfs[wf_idx].windowedWf)

      if not self.detector.IsInDetector(r, 0.1, z):
        return self.draw_position(wf_idx)
      else:
        return (r,z, scale, None)

    def from_prior(self):
        priors = self.priors
        prior_vars = self.prior_vars
        detector=self.detector

        num_waveforms = self.num_waveforms
        rad_arr    = np.empty(num_waveforms)
        phi_arr    = np.empty(num_waveforms)
        theta_arr  = np.empty(num_waveforms)
        scale_arr  = np.empty(num_waveforms)
        t0_arr     = np.empty(num_waveforms)
        smooth_arr = np.empty(num_waveforms)

        #draw waveform params for each waveform
        for (wf_idx) in range(num_waveforms):
            (r,z, scale, t0) = self.draw_position(wf_idx)
            rad = np.sqrt(r**2+z**2)
            theta = np.arctan(z/r)
            smooth_guess = 10

            rad_arr[wf_idx] = rad
            phi_arr[wf_idx] = rng.rand() * np.pi/4
            theta_arr[wf_idx] = theta
            scale_arr[wf_idx] = 5*rng.randn() + scale - .005*scale
            t0_arr[wf_idx] = dnest4.wrap(3*rng.randn() + self.alignidx_guess, self.min_maxt, self.max_maxt)
            smooth_arr[wf_idx] = dnest4.wrap(rng.randn() + smooth_guess, 0, 20)


        #TF params
        phi = dnest4.wrap(prior_vars[phi_idx]*rng.randn() + priors[phi_idx], -np.pi/2, np.pi/2)
        omega = dnest4.wrap(prior_vars[omega_idx]*rng.randn() + priors[omega_idx], 0, np.pi)
        d = dnest4.wrap(prior_vars[d_idx]*rng.randn() + priors[d_idx], 0.5, 1)

        #RC params are normal & clipped
        rc1 = dnest4.wrap(prior_vars[rc1_idx]*rng.randn() + priors[rc1_idx], 65, 80)
        rc2 = dnest4.wrap(prior_vars[rc2_idx]*rng.randn() + priors[rc2_idx], 0.1, 10)
        rcfrac = dnest4.wrap(prior_vars[rcfrac_idx]*rng.randn() + priors[rcfrac_idx], 0.9, 1)

        # aliasrc = dnest4.wrap(prior_vars[aliasrc_idx]*rng.randn() + priors[aliasrc_idx], 0.01, 10)
        # aliasrc = rng.rand()*(10 - 0.01) +  0.01

        h_100_va = dnest4.wrap(prior_vars[velo_first_idx]*rng.randn() + priors[velo_first_idx], 1, 10*priors[velo_first_idx])
        h_111_va = dnest4.wrap(prior_vars[velo_first_idx+1]*rng.randn() + priors[velo_first_idx+1], 1, 10*priors[velo_first_idx])
        h_100_vmax = dnest4.wrap(prior_vars[velo_first_idx+2]*rng.randn() + priors[velo_first_idx+2], 1, 10*priors[velo_first_idx])
        h_111_vmax = dnest4.wrap(prior_vars[velo_first_idx+3]*rng.randn() + priors[velo_first_idx+3], 1, 10*priors[velo_first_idx])
        h_100_beta = (self.conf.beta_lims[1] - self.conf.beta_lims[0]) * rng.rand() + self.conf.beta_lims[0]
        h_111_beta = (self.conf.beta_lims[1] - self.conf.beta_lims[0]) * rng.rand() + self.conf.beta_lims[0]

        # grad = dnest4.wrap(prior_vars[grad_idx]*rng.randn() + priors[grad_idx], detector.gradList[0], detector.gradList[-1])
        # avgImp = dnest4.wrap(prior_vars[imp_avg_idx]*rng.randn() + priors[imp_avg_idx], detector.impAvgList[0], detector.impAvgList[-1])

        grad = rng.rand()*(detector.gradList[-1] - detector.gradList[0]) +  detector.gradList[0]
        avgImp = rng.rand()*(detector.impAvgList[-1] - detector.impAvgList[0]) +  detector.impAvgList[0]
        pcRad = rng.rand()*(detector.pcRadList[-1] - detector.pcRadList[0]) +  detector.pcRadList[0]
        pcLen = rng.rand()*(detector.pcLenList[-1] - detector.pcLenList[0]) +  detector.pcLenList[0]

        #uniform random for charge trapping
        charge_trapping = rng.rand()*(5000 - self.conf.traprc_min) +  self.conf.traprc_min

        return np.hstack([
              phi, omega, d,
              rc1, rc2, rcfrac,#aliasrc,
              h_100_va, h_111_va, h_100_vmax, h_111_vmax, h_100_beta, h_111_beta,
              grad, avgImp, pcRad, pcLen, charge_trapping,
              rad_arr[:], phi_arr[:], theta_arr[:], scale_arr[:], t0_arr[:],smooth_arr[:],
            ])

    def perturb(self, params):

        logH = 0.0
        num_waveforms = self.num_waveforms

        #decide whether to adjust just waveforms, or both wfs and detector params

        if rng.rand() <= 0.5:
            #adjust detector only
            reps = 1;
            if(rng.rand() < 0.5):
                reps += np.int(np.power(100.0, rng.rand()));

            for i in range(reps):
                which = rng.randint(len(self.priors))
                logH += self.perturb_detector(params, which)
        else:
            #adjust at least one waveform:
            self.changed_wfs.fill(0)
            randt2 = rng.randn()/np.sqrt(-np.log(rng.rand()));
            chance = np.power(10.0, -3*np.abs(randt2));

            for wf_idx in range(num_waveforms):
                if rng.rand() <= chance:
                     self.changed_wfs[wf_idx] = True
                #make sure one waveform is changed:
            if np.any(self.changed_wfs) == 0:
                self.changed_wfs[rng.randint(num_waveforms)] = 1

            for wf_idx in range(num_waveforms):
                if self.changed_wfs[wf_idx] == 1:
                    logH += self.perturb_wf(params, wf_idx)
        return logH

    def perturb_detector(self, params, which):
        logH = 0.0
        detector = self.detector
        num_waveforms =  self.num_waveforms
        priors = self.priors
        prior_vars = self.prior_vars

        if which == phi_idx: #lets call this phi
            logH -= -0.5*((params[which] - priors[which])/prior_vars[which])**2
            params[which] += prior_vars[which]*dnest4.randh()
            params[which] = dnest4.wrap(params[which], -np.pi/2, np.pi/2)
            logH += -0.5*((params[which] - priors[which])/prior_vars[which])**2
        elif which == omega_idx: #call it omega
            logH -= -0.5*((params[which] - priors[which])/prior_vars[which])**2
            params[which] += prior_vars[which]*dnest4.randh()
            params[which] = dnest4.wrap(params[which], 0, np.pi)
            logH += -0.5*((params[which] - priors[which])/prior_vars[which])**2
        elif which == d_idx: #d
            logH -= -0.5*((params[which] - priors[which])/prior_vars[which])**2
            params[which] += prior_vars[which]*dnest4.randh()
            params[which] = dnest4.wrap(params[which], 0.5, 1)
            logH += -0.5*((params[which] - priors[which])/prior_vars[which])**2

        elif which == rc1_idx or which == rc2_idx or which == rcfrac_idx:
            #all normally distributed priors
            logH -= -0.5*((params[which] - priors[which])/prior_vars[which])**2
            params[which] += prior_vars[which]*dnest4.randh()
            logH += -0.5*((params[which] - priors[which])/prior_vars[which])**2

        # elif which == aliasrc_idx:
        #     params[which] += 19.9*dnest4.randh()
        #     params[which] = dnest4.wrap(params[which], 0.1, 20)

        elif which == grad_idx:
            # logH -= -0.5*((params[which] - priors[which])/prior_vars[which])**2
            # params[which] += prior_vars[which] *dnest4.randh()
            # params[which] = dnest4.wrap(params[which], detector.gradList[0], detector.gradList[-1])
            # logH += -0.5*((params[which] - priors[which])/prior_vars[which])**2

            paramlist = detector.gradList
            params[which] = (paramlist[-1] - paramlist[0])*dnest4.randh()
            params[which] = dnest4.wrap(params[which], paramlist[0], paramlist[-1])

        elif which == imp_avg_idx:
            # logH -= -0.5*((params[which] - priors[which])/prior_vars[which])**2
            # params[which] += prior_vars[which] *dnest4.randh()
            # params[which] = dnest4.wrap(params[which], detector.impAvgList[0], detector.impAvgList[-1])
            # logH += -0.5*((params[which] - priors[which])/prior_vars[which])**2
            paramlist = detector.impAvgList
            params[which] = (paramlist[-1] - paramlist[0])*dnest4.randh()
            params[which] = dnest4.wrap(params[which], paramlist[0], paramlist[-1])

        elif which == pcrad_idx:
            paramlist = detector.pcRadList
            params[which] = (paramlist[-1] - paramlist[0])*dnest4.randh()
            params[which] = dnest4.wrap(params[which], paramlist[0], paramlist[-1])

        elif which == pclen_idx:
            paramlist = detector.pcLenList
            params[which] = (paramlist[-1] - paramlist[0])*dnest4.randh()
            params[which] = dnest4.wrap(params[which], paramlist[0], paramlist[-1])

        elif which == trap_idx:
            params[which] += (5000 - self.conf.traprc_min)*dnest4.randh()
            params[which] = dnest4.wrap(params[which], self.conf.traprc_min, 5000)

        elif which >= velo_first_idx and which < velo_first_idx+4:
            logH -= -0.5*((params[which] - priors[which])/prior_vars[which])**2
            params[which] += prior_vars[which]*dnest4.randh()
            params[which] = dnest4.wrap(params[which], 1, priors[which]*10)
            logH += -0.5*((params[which] - priors[which])/prior_vars[which])**2

        elif which == velo_first_idx+4 or which == velo_first_idx+5:
            params[which] += (self.conf.beta_lims[1] - self.conf.beta_lims[0])  *dnest4.randh()
            params[which] = dnest4.wrap(params[which], self.conf.beta_lims[0], self.conf.beta_lims[1])

        else: #velocity or rc params: cant be below 0, can be arb. large
            print ("which value %d not supported" % which)
            exit(0)

        return logH


    def perturb_wf(self, params, wf_idx, ):
    #do both wf and detector params in case theres strong correlation
        logH = 0.0
        num_waveforms = self.num_waveforms
        detector = self.detector
        priors = self.priors

        reps = 1
        if rng.rand() < 0.5:
            reps += np.int(np.power(100.0, rng.rand()));

        for i in range(reps):
            wf_which = rng.randint(6)

            # my_which = rng.randint(len(priors) + 8)

            # if my_which < len(priors):
            #     #detector variable
            #     logH += self.perturb_detector(params, my_which)
            #
            # else:
            if wf_which < 6:
                #this is a waveform variable!
                # wf_which =  np.int(my_which - len(priors))

                #which idx of the global params array
                which = len(priors) + wf_which*num_waveforms + wf_idx

                rad_idx = len(priors) + wf_idx
                theta_idx =  len(priors) + 2*num_waveforms+ wf_idx
                self.changed_wfs[wf_idx] = 1

                if wf_which == 0:
                  theta = params[theta_idx]

                  IsInDetector = 0
                  while not(IsInDetector):
                    new_rad = self.get_new_rad(params[which], theta)
                    r = new_rad * np.cos(theta)
                    z = new_rad * np.sin(theta)
                    IsInDetector = detector.IsInDetector(r, 0,z)

                  params[which] = new_rad

                elif wf_which ==2: #theta
                    rad = params[rad_idx]

                    IsInDetector = 0
                    while not(IsInDetector):
                      new_theta = self.get_new_theta(rad, params[which])
                      r = rad * np.cos(new_theta)
                      z = rad * np.sin(new_theta)
                      IsInDetector = detector.IsInDetector(r, 0,z)
                    params[which] = new_theta

                # if wf_which == 0:
                #     params[which] += (detector.detector_radius)*dnest4.randh()
                #     params[which] = dnest4.wrap(params[which] , 0, detector.detector_radius)
                elif wf_which == 1:
                    max_val = np.pi/4
                    params[which] += np.pi/4*dnest4.randh()
                    params[which] = dnest4.wrap(params[which], 0, max_val)

                elif wf_which == 3: #scale
                    wf = self.wfs[wf_idx]
                    wf_guess = wf.wfMax

                    sig = 20

                    logH -= -0.5*((params[which] - wf.wfMax  )/sig)**2
                    params[which] += sig*dnest4.randh()
                    params[which] = dnest4.wrap(params[which], 0.8*wf_guess, 1.2*wf_guess)
                    logH += -0.5*((params[which] - wf.wfMax )/sig)**2

                elif wf_which == 4: #t0
                  #gaussian around 0, sigma... 5?
                  t0_sig = self.maxt_sigma
                  logH -= -0.5*((params[which] - self.alignidx_guess )/t0_sig)**2
                  params[which] += t0_sig*dnest4.randh()
                  params[which] = dnest4.wrap(params[which], self.min_maxt, self.max_maxt)
                  logH += -0.5*((params[which] - self.alignidx_guess)/t0_sig)**2

                elif wf_which == 5: #smooth
                  #gaussian around 10
                  sig = 3
                  logH -= -0.5*((params[which] - 10 )/sig)**2
                  params[which] += sig*dnest4.randh()
                  params[which] = dnest4.wrap(params[which], 0, 20)
                  logH += -0.5*((params[which] - 10)/sig)**2

                else:
                    print( "wf which value %d (which value %d) not supported" % (wf_which, which) )
                    exit(0)

        return logH

    def get_new_rad(self,rad, theta):
          detector = self.detector
          #FIND THE MAXIMUM RADIUS STILL INSIDE THE DETECTOR
          theta_eq = np.arctan(detector.detector_length/detector.detector_radius)
          theta_taper = np.arctan(detector.taper_length/detector.detector_radius)
          if theta <= theta_taper:
             z = np.tan(theta)*(detector.detector_radius - detector.taper_length) / (1-np.tan(theta))
             max_rad = z / np.sin(theta)
          elif theta <= theta_eq:
              max_rad = detector.detector_radius / np.cos(theta)
          else:
              theta_comp = np.pi/2 - theta
              max_rad = detector.detector_length / np.cos(theta_comp)

          #AND THE MINIMUM (from PC dimple)
          #min_rad  = 1./ ( np.cos(theta)**2/detector.pcRad**2  +  np.sin(theta)**2/detector.pcLen**2 )

          min_rad = np.amax([detector.pcRad, detector.pcLen])

          new_rad = rad + (max_rad - min_rad)*dnest4.randh()
          new_rad = dnest4.wrap(new_rad, min_rad, max_rad)
          return new_rad
    def get_new_theta(self,rad,theta):
        detector = self.detector
        if rad < np.amin([detector.detector_radius - detector.taper_length, detector.detector_length]):
            max_val = np.pi/2
            min_val = 0
        else:
            if rad < detector.detector_radius - detector.taper_length:
                #can't possibly hit the taper
                min_val = 0
            elif rad < np.sqrt(detector.detector_radius**2 + detector.taper_length**2):
                #low enough that it could hit the taper region
                a = detector.detector_radius - detector.taper_length
                z = 0.5 * (np.sqrt(2*rad**2-a**2) - a)
                min_val = np.arcsin(z/rad)
            else:
                #longer than could hit the taper
                min_val = np.arccos(detector.detector_radius/rad)

            if rad < detector.detector_length:
                max_val = np.pi/2
            else:
                max_val = np.pi/2 - np.arccos(detector.detector_length/rad)

        new_theta = theta + (max_val - min_val)*dnest4.randh()
        new_theta = dnest4.wrap(new_theta, min_val, max_val)
        return new_theta

    def log_likelihood(self, params):
        return self.fit_manager.calc_likelihood(params)

    def get_indices(self):
        return (tf_first_idx, velo_first_idx, trap_idx, grad_idx, len(priors))

    def calc_wf_likelihood(self, wf_params, wf_idx ):
        wf = self.wfs[wf_idx]
        data = wf.windowedWf
        model_err = 0.57735027 * wf.baselineRMS
        data_len = len(data)
        model = self.make_waveform(data_len, wf_params, )

        if model is None:
            ln_like = -np.inf
        else:
            inv_sigma2 = 1.0/(model_err**2)
            ln_like = -0.5*(np.sum((data-model)**2*inv_sigma2 - np.log(inv_sigma2)))

        return ln_like

    def make_waveform(self, data_len, wf_params, charge_type=None):

        # tf_phi, tf_omega, d, rc1, rc2, rcfrac, aliasrc = wf_params[tf_first_idx:tf_first_idx+7]
        tf_phi, tf_omega, d, rc1, rc2, rcfrac,  = wf_params[tf_first_idx:tf_first_idx+6]
        # h_100_va, h_111_va, h_100_vmax, h_111_vmax, h_100_beta, h_111_beta, = wf_params[velo_first_idx:velo_first_idx+6]
        h_100_vlo, h_111_vlo, h_100_vhi, h_111_vhi, h_100_beta, h_111_beta = wf_params[velo_first_idx:velo_first_idx+6]
        charge_trapping = wf_params[trap_idx]
        grad = wf_params[grad_idx]
        avg_imp = wf_params[imp_avg_idx]

        pcrad,pclen = wf_params[pcrad_idx:pcrad_idx+2]

        rad, phi, theta, scale, maxt, smooth =  wf_params[self.num_det_params:]

        r = rad * np.cos(theta)
        z = rad * np.sin(theta)

        c = -d * np.cos(tf_omega)
        b_ov_a = c - np.tan(tf_phi) * np.sqrt(d**2-c**2)
        a = 1./(1+b_ov_a)
        tf_b = a * b_ov_a

        h_100_mu0, h_100_beta, h_100_e0 = self.get_velo_params(h_100_vlo, h_100_vhi, h_100_beta)
        h_111_mu0, h_111_beta, h_111_e0 = self.get_velo_params(h_111_vlo, h_111_vhi, h_111_beta)

        if scale < 0:
            return None
        if smooth < 0:
            return None
        if not self.detector.IsInDetector(r, phi, z):
            return None

        self.detector.SetTransferFunction(tf_b, c, d, rc1, rc2, rcfrac, )
        self.detector.siggenInst.set_hole_params(h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0)
        self.detector.trapping_rc = charge_trapping
        # self.detector.SetAntialiasingRC(aliasrc)
        self.detector.rc_int_exp = None
        self.detector.SetGrads(grad, avg_imp)
        self.detector.SetPointContact(pcrad, pclen)

        if charge_type is None:
            if self.conf.alignType == "max":
                model = self.detector.MakeSimWaveform(r, phi, z, scale, maxt, data_len, h_smoothing=smooth, alignPoint="max", doMaxInterp=self.conf.doMaxInterp)
            elif self.conf.alignType == "timepoint":
                model = self.detector.MakeSimWaveform(r, phi, z, scale, maxt, data_len, h_smoothing=smooth, alignPoint= self.conf.align_percent)
        elif charge_type == 1:
            model = self.detector.MakeRawSiggenWaveform(r, phi, z,1)
        elif charge_type == -1:
            model = self.detector.MakeRawSiggenWaveform(r, phi, z,-1)

        else:
            print("Not a valid charge type! {0}".format(charge_type))

        if model is None or np.any(np.isnan(model)):
            return None

        return model
    #
    # def get_velo_params(self, v_a, v_max, beta):
    #     E_a = self.conf.E_a
    #     E_0 = np.power( (v_max*E_a/v_a)**beta - E_a**beta , 1./beta)
    #     mu_0 = v_max / E_0
    #
    #     return (mu_0,  beta, E_0)

    def get_velo_params(self, v_a, v_c, beta):
        E_a = self.conf.E_lo
        E_c = self.conf.E_hi

        # beta = 1./np.exp(logb)

        psi = (E_a * v_c) / ( E_c * v_a )
        E_0 = np.power((psi**beta* E_c**beta - E_a**beta) / (1-psi**beta), 1./beta)
        mu_0 = (v_a / E_a) * (1 + (E_a / E_0)**beta )**(1./beta)

        return (mu_0,  beta, E_0)


    def get_indices(self):
        return (tf_first_idx, velo_first_idx, grad_idx, trap_idx)
