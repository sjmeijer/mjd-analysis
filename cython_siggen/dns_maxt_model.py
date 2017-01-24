import numpy as np
import scipy.stats as stats
import scipy.optimize as op
import dnest4

import numpy.random as rng
from multiprocessing import Pool

def initializeDetector(det, reinit=True):
  global detector
  detector = det
  # if reinit:
  #     detector.ReinitializeDetector

def initializeWaveforms( wfs_init, ):
  global wfs
  wfs = wfs_init

  global num_waveforms
  num_waveforms = len(wfs)


def initializeDetectorAndWaveforms(det, wfs_init, reinit=True):
  initializeWaveforms(wfs_init)
  initializeDetector(det, reinit)

def initMultiThreading(numThreads):
  global num_threads
  global pool
  num_threads = numThreads
  if num_threads > 1:
      pool = Pool(num_threads, initializer=initializeDetector, initargs=[detector])

def initT0Padding(maxt_pad, linear_baseline_origin):
    global maxt_guess, min_maxt, max_maxt, baseline_origin_idx
    maxt_guess = maxt_pad
    max_maxt = maxt_guess + 10
    min_maxt = maxt_guess - 10
    baseline_origin_idx = linear_baseline_origin

traprc_min = 100

tf_first_idx = 0
velo_first_idx = 6
trap_idx = 13
grad_idx = 12

priors = np.empty(6 + 6 + 1 + 1) #6 + 2)

ba_idx, c_idx, dc_idx = np.arange(3)+ tf_first_idx
rc1_idx, rc2_idx, rcfrac_idx = np.arange(3)+ tf_first_idx+3

#3 transfer function params for oscillatory decay
ba_prior = 0.107213
c_prior = -0.815152
dc_prior = 0.822696/-0.815152

rc1_prior =  73.085166
rc2_prior = 1.138420
rc_frac_prior = 0.997114

h_100_mu0_prior, h_100_beta_prior, h_100_e0_prior = 66333., 0.744, 181.
h_111_mu0_prior, h_111_beta_prior, h_111_e0_prior =  107270., 0.580, 100.

priors[tf_first_idx:tf_first_idx+3] = ba_prior, c_prior, dc_prior
priors[rc1_idx:rc1_idx+3] = rc1_prior, rc2_prior, rc_frac_prior
priors[velo_first_idx:velo_first_idx+3] = h_100_mu0_prior, h_100_beta_prior, h_100_e0_prior
priors[velo_first_idx+3:velo_first_idx+6] = h_111_mu0_prior, h_111_beta_prior, h_111_e0_prior

prior_vars =  np.empty(len(priors))
prior_vars[rc1_idx:rc1_idx+3] = 0.2, 0.3, 0.001

velo_width = 10.
velo_var = 1.
prior_vars[velo_first_idx:velo_first_idx+6] = velo_var*priors[velo_first_idx:velo_first_idx+6]

priors[grad_idx] = 100
prior_vars[grad_idx] = 3
priors[trap_idx] = 200.
prior_vars[trap_idx] = 25.

def get_velo_params():
    return (priors[velo_first_idx:velo_first_idx+6], velo_var)
def get_t0_params():
    return (t0_guess, min_t0, max_t0, )
def get_param_idxs():
    return (tf_first_idx, velo_first_idx, grad_idx, trap_idx)

def draw_position(wf_idx):
  r = rng.rand() * detector.detector_radius
  z = rng.rand() * detector.detector_radius
  scale = np.amax(wfs[wf_idx].windowedWf)
  t0 = None

  if not detector.IsInDetector(r, 0.1, z):
    return draw_position(wf_idx)
  else:
    return (r,z, scale, t0)


class Model(object):
    """
    Specify the model in Python.
    """
    def __init__(self):
        """
        Parameter values *are not* stored inside the class
        """
        changed_wfs = np.zeros(num_waveforms)

    def from_prior(self):
        """
        Unlike in C++, this must *return* a numpy array of parameters.
        """
        r_arr      = np.empty(num_waveforms)
        z_arr      = np.empty(num_waveforms)

        rad_arr      = np.empty(num_waveforms)
        phi_arr    = np.empty(num_waveforms)
        theta_arr  = np.empty(num_waveforms)
        scale_arr  = np.empty(num_waveforms)
        t0_arr     = np.empty(num_waveforms)
        smooth_arr = np.empty(num_waveforms)
        m_arr      = np.empty(num_waveforms)
        b_arr      = np.empty(num_waveforms)

        print "\n"
        #draw 8 waveform params for each waveform
        for (wf_idx, wf) in enumerate(wfs):
            (r,z, scale, t0) = draw_position(wf_idx)
            smooth_guess = 10

            r_arr[wf_idx] = r
            z_arr[wf_idx] = z
            # rad_arr[wf_idx] = rad
            phi_arr[wf_idx] = rng.rand() * np.pi/4
            # theta_arr[wf_idx] = theta
            scale_arr[wf_idx] = 5*rng.randn() + scale - .005*scale
            t0_arr[wf_idx] = 3*rng.randn() + maxt_guess
            smooth_arr[wf_idx] = np.clip(rng.randn() + smooth_guess, 0, 20)
            m_arr[wf_idx] =  0.0001*rng.randn() + 0.
            b_arr[wf_idx] =  0.001*rng.randn() + 0.

            # print "  creating wf %d" % wf_idx
            # print "  >>",
            # print rad_arr[wf_idx], phi_arr[wf_idx]/np.pi, theta_arr[wf_idx]/np.pi, t0_arr[wf_idx]
            # print "  ", rad_arr[wf_idx], theta_arr[wf_idx]/np.pi

        b_over_a = rng.rand() * 400 - 100
        c = 0.05 *rng.randn() + c_prior
        dc =  0.01 *rng.randn() + dc_prior

        #limit from 60 to 90
        rc1 = dnest4.wrap(prior_vars[rc1_idx]*rng.randn() + priors[rc1_idx], 65, 80)
        rc2 = dnest4.wrap(prior_vars[rc2_idx]*rng.randn() + priors[rc2_idx], 0.1, 10)
        rcfrac = dnest4.wrap(prior_vars[rcfrac_idx]*rng.randn() + priors[rcfrac_idx], 0.9, 1)
        charge_trapping = dnest4.wrap(prior_vars[trap_idx]*rng.randn() + priors[trap_idx], 50, 1000)

        grad = np.int(np.clip(prior_vars[grad_idx]*np.int(rng.randn()) + priors[grad_idx], 0, len(detector.gradList)-1))

        #6 hole drift params
        h_100_mu0 = .01 * h_100_mu0_prior*rng.randn() + h_100_mu0_prior
        lnbeta = np.log(1/(.1*h_100_beta_prior))*rng.randn() + np.log(1./h_100_beta_prior)
        h_100_lnbeta = dnest4.wrap(lnbeta, 0, np.log(1/.1))
        h_100_emu = .01 * (h_100_e0_prior*h_100_mu0_prior)*rng.randn() + h_100_e0_prior*h_100_mu0_prior

        h_111_mu0 = .01 * h_111_mu0_prior*rng.randn() + h_111_mu0_prior
        lnbeta = np.log(1/(.1*h_111_beta_prior))*rng.randn() + np.log(1./h_111_e0_prior)
        h_111_lnbeta = dnest4.wrap(lnbeta, 0, np.log(1/.1))
        h_111_emu = .01 * (h_111_e0_prior*h_111_mu0_prior)*rng.randn() + h_111_e0_prior*h_111_mu0_prior

        return np.hstack([
              b_over_a, c, dc,
              rc1, rc2, rcfrac,
              h_100_mu0, h_100_lnbeta, h_100_emu, h_111_mu0, h_111_lnbeta, h_111_emu,
              grad, charge_trapping,
              r_arr[:], phi_arr[:], z_arr[:], scale_arr[:], t0_arr[:],smooth_arr[:], m_arr[:], b_arr[:]
            ])

    def perturb(self, params):
        """
        Unlike in C++, this takes a numpy array of parameters as input,
        and modifies it in-place. The return value is still logH.
        """
        logH = 0.0
        which = rng.randint(len(params))

        if which >= len(priors):
            #this is a waveform variable!
            wf_which = np.floor((which - len(priors)) / num_waveforms)

            if wf_which == 0:
                params[which] += (detector.detector_radius)*dnest4.randh()
                params[which] = dnest4.wrap(params[which] , 0, detector.detector_radius)
            elif wf_which == 1:
                max_val = np.pi/4
                params[which] += np.pi/4*dnest4.randh()
                params[which] = dnest4.wrap(params[which], 0, max_val)
                if params[which] < 0 or params[which] > np.pi/4:
                    print "wtf phi"
                #params[which] = np.clip(params[which], 0, max_val)
            elif wf_which == 2:
                params[which] += (detector.detector_length)*dnest4.randh()
                params[which] = dnest4.wrap(params[which] , 0, detector.detector_length)

            elif wf_which == 3: #scale
                wf_idx = (which - len(priors)) % num_waveforms
                wf = wfs[wf_idx]
                min_scale = wf.wfMax - 0.01*wf.wfMax
                max_scale = wf.wfMax + 0.005*wf.wfMax
                params[which] += (max_scale-min_scale)*dnest4.randh()
                params[which] = dnest4.wrap(params[which], min_scale, max_scale)
            #   print "  adjusted scale to %f" %  ( params[which])

            elif wf_which == 4: #t0
              params[which] += 1*dnest4.randh()
              params[which] = dnest4.wrap(params[which], min_maxt, max_maxt)
            elif wf_which == 5: #smooth
              params[which] += 0.1*dnest4.randh()
              params[which] = dnest4.wrap(params[which], 0, 25)
            #   print "  adjusted smooth to %f" %  ( params[which])

            elif wf_which == 6: #wf baseline slope (m)
                logH -= -0.5*(params[which]/1E-4)**2
                params[which] += 1E-4*dnest4.randh()
                logH += -0.5*(params[which]/1E-4)**2
            elif wf_which == 7: #wf baseline incercept (b)
                logH -= -0.5*(params[which]/1E-2)**2
                params[which] += 1E-2*dnest4.randh()
                logH += -0.5*(params[which]/1E-2)**2

            #   params[which] += 0.01*dnest4.randh()
            #   params[which]=dnest4.wrap(params[which], -1, 1)
            #   print "  adjusted b to %f" %  ( params[which])

        elif which == ba_idx: #b over a
            params[which] += 600*dnest4.randh()
            params[which] = dnest4.wrap(params[which], -200, 400)
        elif which == c_idx: #b over a
            params[which] += 0.01*dnest4.randh()
            params[which] = dnest4.wrap(params[which], -0.9, -0.7)
        elif which == dc_idx: #b over a
            params[which] += 0.01*dnest4.randh()
            params[which] = dnest4.wrap(params[which], -1.05, -0.975)

        elif which == rc1_idx or which == rc2_idx or which == rcfrac_idx:
            #all normally distributed priors
            logH -= -0.5*((params[which] - priors[which])/prior_vars[which])**2
            params[which] += prior_vars[which]*dnest4.randh()
            logH += -0.5*((params[which] - priors[which])/prior_vars[which])**2

        elif which == grad_idx:
          params[which] += (len(detector.gradList)-1)*dnest4.randh()
          params[which] = np.int(dnest4.wrap(params[which], 0, len(detector.gradList)-1))
        elif which >= velo_first_idx and which < velo_first_idx+6:
            velo_which =  (which - velo_first_idx)%3
            #TODO: consider long transforming priors on two of these
            if velo_which ==0: #mu0 parameter
                params[which] += (500E3 - 10E3)  *dnest4.randh()
                params[which] = dnest4.wrap(params[which], 10E3, 500E3)
            elif velo_which ==1: #ln(1/beta)
                space = np.log(1/.1)
                params[which] += space *dnest4.randh()
                params[which] = dnest4.wrap(params[which], 0, np.log(1/.1))
            elif velo_which == 2:
                space = 10E3 * 100 - 500E3 * 300
                params[which] += space *dnest4.randh()
                params[which] = dnest4.wrap(params[which], 10E3 * 100, 500E3 * 300)

        elif which == trap_idx:
          params[which] += prior_vars[trap_idx]*dnest4.randh()
          params[which] = dnest4.wrap(params[which], 50, 1000)

        else: #velocity or rc params: cant be below 0, can be arb. large
            print "which value %d not supported" % which
            exit(0)


        return logH

    def log_likelihood(self, params):
        """
        Gaussian sampling distribution.
        """
        b_over_a, c, dc, rc1, rc2, rcfrac = params[tf_first_idx:tf_first_idx+6]
        h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0 = params[velo_first_idx:velo_first_idx+6]
        charge_trapping = params[trap_idx]
        grad = np.int(params[grad_idx])

        rad_arr, phi_arr, theta_arr, scale_arr, t0_arr, smooth_arr, m_arr, b_arr = params[len(priors):].reshape((8, num_waveforms))
        sum_like = 0
        if num_threads > 1:
            args = []
            # sum_like = 0
            for (wf_idx, wf) in enumerate(wfs):
                # print rad_arr[wf_idx]
                args.append([wf,  rad_arr[wf_idx], phi_arr[wf_idx], theta_arr[wf_idx],
                              scale_arr[wf_idx], t0_arr[wf_idx], smooth_arr[wf_idx],
                              m_arr[wf_idx], b_arr[wf_idx],
                              b_over_a, c, dc, rc1, rc2, rcfrac,
                              h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0,
                              grad, charge_trapping, baseline_origin_idx
                            ])

            results = pool.map(WaveformLogLikeStar, args)
            sum_like = np.sum(results)
        else:
            for (wf_idx, wf) in enumerate(wfs):
                sum_like += WaveformLogLike(wf,  rad_arr[wf_idx], phi_arr[wf_idx], theta_arr[wf_idx],
                              scale_arr[wf_idx], t0_arr[wf_idx], smooth_arr[wf_idx],
                              m_arr[wf_idx], b_arr[wf_idx],
                              b_over_a, c, dc, rc1, rc2, rcfrac,
                              h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0,
                              grad, charge_trapping, baseline_origin_idx
                            )

        return sum_like


def WaveformLogLikeStar(a_b):
  return WaveformLogLike(*a_b)

def WaveformLogLike(wf, r, phi, z, scale, maxt, smooth, m, b, b_over_a, c, dc, rc1, rc2, rcfrac,
        h_100_mu0, h_100_lnbeta, h_100_emu, h_111_mu0, h_111_lnbeta, h_111_emu,
        grad, charge_trapping, bl_origin_idx):
    # #TODO: This needs to be length normalized somehow
    # print "think about length normalization, you damn fool"
    # exit(0)

    # print "theta is %f" % (theta/np.pi)
    # r = rad * np.cos(theta)
    # z = rad * np.sin(theta)
    d = c * dc

    # rc1 = -1./np.log(e_rc1)
    # rc2 = -1./np.log(e_rc2)
    # charge_trapping = -1./np.log(e_charge_trapping)

    h_100_beta = 1./np.exp(h_100_lnbeta)
    h_111_beta = 1./np.exp(h_111_lnbeta)
    h_100_e0 = h_100_emu / h_100_mu0
    h_111_e0 = h_111_emu / h_111_mu0

    if scale < 0:
      return -np.inf
    if smooth < 0:
       return -np.inf
    if not detector.IsInDetector(r, phi, z):
      return -np.inf

    detector.SetTransferFunction(b_over_a, c, d, rc1, rc2, rcfrac)
    detector.siggenInst.set_hole_params(h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0)
    detector.trapping_rc = charge_trapping
    detector.SetFieldsGradIdx(grad)

    data = wf.windowedWf
    model_err = wf.baselineRMS
    data_len = len(data)

    model = detector.MakeSimWaveform(r, phi, z, scale, maxt, data_len, h_smoothing=smooth, alignPoint="max")
    if model is None:
      return -np.inf
    if np.any(np.isnan(model)): return -np.inf

    start_idx = -bl_origin_idx
    end_idx = data_len - bl_origin_idx - 1
    baseline_trend = np.linspace(m*start_idx+b, m*end_idx+b, data_len)
    model += baseline_trend

    inv_sigma2 = 1.0/(model_err**2)
    ln_like = -0.5*(np.sum((data-model)**2*inv_sigma2 - np.log(inv_sigma2)))
    return ln_like

def findTimePointBeforeMax(data, percent):

  #don't screw up the data, bro
  int_data = np.copy(data)
  max_idx = np.argmax(int_data)
  int_data /= int_data[max_idx]

  int_data = int_data[0:max_idx]
  try:
      return np.where(np.less(int_data, percent))[0][-1]
  except IndexError:
      return 0
