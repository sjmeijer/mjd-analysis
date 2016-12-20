import numpy as np
import scipy.stats as stats
import scipy.optimize as op
import dnest4

import numpy.random as rng
from multiprocessing import Pool

def initializeDetector(det, reinit=True):
  global detector
  detector = det
  if reinit:
      detector.ReinitializeDetector

def initializeWaveforms( wfs_init, wfs_guess_result):
  global wfs
  wfs = wfs_init

  global num_waveforms
  num_waveforms = len(wfs)

  global wf_guesses
  wf_guesses = wfs_guess_result


def initializeDetectorAndWaveforms(det, wfs_init, wf_guess_init, reinit=True):
  initializeWaveforms(wfs_init, wf_guess_init)
  initializeDetector(det, reinit)

def initMultiThreading(numThreads):
  global pool
  pool = Pool(numThreads, initializer=initializeDetector, initargs=[detector])

max_t0 = 105
min_t0 = 80

tf_first_idx = 0
velo_first_idx = 6
trap_idx = 12
grad_idx = 13

priors = np.empty(6 + 6 + 2)

ba_idx, c_idx, d_idx = np.arange(3)+ tf_first_idx
rc1_idx, rc2_idx, rcfrac_idx = np.arange(3)+ tf_first_idx+3

#3 transfer function params for oscillatory decay
ba_prior = 0.107213
c_prior = -0.815152
d_prior = 0.822696

rc1_prior = 80.
rc2_prior = 2.08
rc_frac_prior = 0.992

h_100_mu0_prior, h_100_beta_prior, h_100_e0_prior = 66333., 0.744, 181.
h_111_mu0_prior, h_111_beta_prior, h_111_e0_prior =  107270., 0.580, 100.

priors[tf_first_idx:tf_first_idx+3] = ba_prior, c_prior, d_prior
priors[rc1_idx:rc1_idx+3] = rc1_prior, rc2_prior, rc_frac_prior
priors[velo_first_idx:velo_first_idx+3] = h_100_mu0_prior, h_100_beta_prior, h_100_e0_prior
priors[velo_first_idx+3:velo_first_idx+6] = h_111_mu0_prior, h_111_beta_prior, h_111_e0_prior

priors[trap_idx] = 120.

prior_vars =  np.empty(len(priors))

prior_vars[rc1_idx:rc1_idx+3] = 0.05*rc1_prior, 0.05*rc2_prior, 0.001

var = 0.01
prior_vars[velo_first_idx:velo_first_idx+6] = var*priors[velo_first_idx:velo_first_idx+6]
prior_vars[trap_idx] = 1.

priors[grad_idx] = 1
prior_vars[grad_idx] = 1


def draw_position(wf_idx):
#  det_max = np.sqrt(detector.detector_radius**2 + detector.detector_length**2)

#  r = rng.rand() * detector.detector_radius
#  z = rng.rand() * detector.detector_radius

#  number = 100
#  dt_array = np.load("P42574A_drifttimes.npy")
#  r_arr = np.linspace(0, detector.detector_radius, number)
#  z_arr = np.linspace(0, detector.detector_length, number)
#  t_50 = findTimePointBeforeMax(wf.windowedWf, 0.5) - 20
#
#  location_idxs = np.where(np.logical_and(np.greater(dt_array, t_50-10), np.less(dt_array, t_50+10)) )
#
#  guess_idx = rng.randint(len(location_idxs[0]))
#  r = r_arr[location_idxs[0][guess_idx]]
#  z = z_arr[location_idxs[1][guess_idx]]
  wf_guess = wf_guesses[wf_idx]

  r, phi, z, scale, t0, smooth = wf_guess['x'][0:6]
  r += rng.randn()*0.1
  z += rng.randn()*0.1

  if not detector.IsInDetector(r, 0.1, z):
#    print "not in detector..."
    return draw_position(wf_idx)
  else:
    return (r,z, scale, t0, smooth)

def random_position(r, z):
  r_init,z_init = r,z
  r += dnest4.randh()*0.1
  z += dnest4.randh()*0.1

  r = dnest4.wrap(r, 0, detector.detector_radius)
  z = dnest4.wrap(z, 0, detector.detector_length)

  if not detector.IsInDetector(r, 0.1, z):
#    print "not in detector..."
    return random_position(r_init,z_init)
  else:
    return (r,z)

class Model(object):
    """
    Specify the model in Python.
    """
    def __init__(self):
        """
        Parameter values *are not* stored inside the class
        """

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

        #draw 8 waveform params for each waveform
        for (wf_idx, wf) in enumerate(wfs):
            (r,z, scale, t0, smooth) = draw_position(wf_idx)
            smooth_guess = 10
            t0 -= 20 #hack to go from 20 to 100 as t0guess
            t0 += 100
            r_arr[wf_idx] = r
            z_arr[wf_idx] = z
            # rad_arr[wf_idx] = np.sqrt(r**2+z**2)
            phi_arr[wf_idx] = rng.rand() * np.pi/4
            # theta_arr[wf_idx] = np.arctan(z/r)
            scale_arr[wf_idx] = 5*rng.randn() + scale
            t0_arr[wf_idx] = 3*rng.randn() + t0
            smooth_arr[wf_idx] = np.clip(rng.randn() + smooth_guess, 0, 20)
            m_arr[wf_idx] =  0.001*rng.randn() + 0.
            b_arr[wf_idx] =  0.01*rng.randn() + 0.

            print "creating wf %d" % wf_idx
            print "  ",
            print r, phi_arr[wf_idx]/np.pi, z, t0_arr[wf_idx],
            # print "  ", rad_arr[wf_idx], theta_arr[wf_idx]/np.pi

        b_over_a = 0.1*rng.randn() + ba_prior

        mean = [0, 0]
        cov = [[1, -0.99], [-0.99, 1]]
        x, y = np.random.multivariate_normal(mean, cov, 1).T
        c = 0.01*x + c_prior
        d = 0.01*y + d_prior

        rc1 = dnest4.wrap(prior_vars[rc1_idx]*rng.randn() + priors[rc1_idx], 50, 100)
        rc2 = dnest4.wrap(prior_vars[rc2_idx]*rng.randn() + priors[rc2_idx], 0, 5)
        rcfrac = dnest4.wrap(prior_vars[rcfrac_idx]*rng.randn() + priors[rcfrac_idx], 0.9, 1)

        charge_trapping = prior_vars[trap_idx]*rng.randn() + priors[trap_idx]

        grad = np.int(np.clip(prior_vars[trap_idx]*np.int(rng.randn()) + priors[trap_idx], 0, len(detector.gradList)-1))

        #6 hole drift params

        h_100_mu0 = .1*var * h_100_mu0_prior*rng.randn() + h_100_mu0_prior
        h_100_beta = .1*var * h_100_beta_prior*rng.randn() + h_100_beta_prior
        h_100_e0 = .1*var * h_100_e0_prior*rng.randn() + h_100_e0_prior
        h_111_mu0 = .1*var * h_111_mu0_prior*rng.randn() + h_111_mu0_prior
        h_111_beta = .1*var * h_111_beta_prior*rng.randn() + h_111_beta_prior
        h_111_e0 = .1*var * h_111_e0_prior*rng.randn() + h_111_e0_prior

        return np.hstack([
              b_over_a, c, d,
              rc1, rc2, rcfrac,
              h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0,
              charge_trapping, grad,
              r_arr[:], phi_arr[:], z_arr[:], scale_arr[:], t0_arr[:],smooth_arr[:], m_arr[:], b_arr[:]
            #   rad_arr[:], phi_arr[:], theta_arr[:], scale_arr[:], t0_arr[:],smooth_arr[:], m_arr[:], b_arr[:]
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
            # print "which idx is %d, value is %f" % (which, params[which])
            # print "  wf which is %d" % wf_which

            if wf_which == 0 or wf_which == 2: #radius and z
              wf_idx = (which - len(priors)) % num_waveforms
              rad_idx = len(priors) + wf_idx
              z_idx =  len(priors) + 2*num_waveforms+ wf_idx

              r, z = random_position(params[rad_idx],params[z_idx])
              params[rad_idx] = r
              params[z_idx] = z

            #   t0_idx =  len(priors) + 4*num_waveforms+ wf_idx
            #   print "  wf_idx is %d" % wf_idx
            #   print "  rad_idx is %d" % rad_idx
            #   print "  t0_idx is %d" % t0_idx

            #   max_rad = np.sqrt(detector.detector_radius**2 + detector.detector_length**2)
              #
            #   mean = [0, 0]
            #   cov = [[1, -0.8], [-0.8, 1]]
            #   x, y = np.random.multivariate_normal(mean, cov, 1).T
            #   r0 = params[rad_idx]
            #   t00 = params[t0_idx]
            #   params[rad_idx] = np.clip(params[rad_idx] +x * 0.1, 0, max_rad)
            #   params[t0_idx] = np.clip(params[t0_idx] +y * 0.1, min_t0, max_t0)
            #   print "  adjusted rad from %f to %f" %  (r0, params[rad_idx])
            #   print "  adjusted t0 from %f to %f" %  (t00, params[t0_idx])
            elif wf_which == 4:
              params[which] += 0.1*dnest4.randh()
              params[which] = dnest4.wrap(params[which], min_t0, max_t0)
              params[which] = np.clip(params[which], min_t0, max_t0)
            elif wf_which == 1 or wf_which == 2: #phi & theta
              if wf_which == 1: max_val = np.pi/4
              if wf_which == 2: max_val = np.pi/2

              params[which] += dnest4.randh()
              params[which] = dnest4.wrap(params[which], 0, max_val)
              params[which] = np.clip(params[which], 0, max_val)

              wf_idx = (which - len(priors)) % num_waveforms
            #   print "  adjusted %d to %f (wf %d)" %  (wf_which, params[which]/np.pi, wf_idx)

            elif wf_which == 3: #scale
              wf_idx = (which - len(priors)) % num_waveforms
              wf = wfs[wf_idx]
              params[which] += dnest4.randh()
              params[which] = dnest4.wrap(params[which], wf.wfMax - 10*wf.baselineRMS, wf.wfMax + 10*wf.baselineRMS)
              params[which] = np.clip(params[which], wf.wfMax - 50*wf.baselineRMS, wf.wfMax + 50*wf.baselineRMS)
            #   print "  adjusted scale to %f" %  ( params[which])
            elif wf_which == 5: #smooth
              params[which] += 0.1*dnest4.randh()
              params[which] = np.clip(params[which], 0, 15)
            #   print "  adjusted smooth to %f" %  ( params[which])

            elif wf_which == 6:
              params[which] += 0.001*dnest4.randh()
              dnest4.wrap(params[which], -0.02, 0.02)
              params[which] = np.clip(params[which], -0.05, 0.05)
            #   print "  adjusted m to %f" %  ( params[which])
            elif wf_which == 7:
              params[which] += 0.01*dnest4.randh()
              dnest4.wrap(params[which], -2, 2)
              params[which] = np.clip(params[which], -5, 5)
            #   print "  adjusted b to %f" %  ( params[which])

        elif which == ba_idx: #b over a
          params[which] += prior_vars[which]*dnest4.randh()
          params[which] += 0.1*dnest4.randh()
          params[which] = dnest4.wrap(params[which], -0.9, 15)

        elif which ==c_idx or which == d_idx: #this is c and d
          mean = [0, 0]
          cov = [[1, -0.99], [-0.99, 1]]
          jumps = np.array((0.1*dnest4.randh(), 0.1*dnest4.randh()))

          (c_jump, d_jump) = np.dot(cov, jumps)

          params[c_idx] += c_jump
          params[d_idx] += d_jump

          params[c_idx] = dnest4.wrap(params[c_idx], -0.9, -0.7 )
          params[d_idx] = dnest4.wrap(params[d_idx],0.7, 0.9)

        #   params[c_idx] = np.clip(params[c_idx], -0.9, -0.7 )
        #   params[d_idx] = np.clip(params[d_idx],0.7, 0.9)

        elif which == rc1_idx:
          params[which] += prior_vars[which]*dnest4.randh()
          params[which] = dnest4.wrap(params[which], 70, 100)
        elif which == rc2_idx:
          params[which] += prior_vars[which]*dnest4.randh()
          params[which] = dnest4.wrap(params[which], 1, 5)
        elif which == rcfrac_idx:
          params[which] += prior_vars[which]*dnest4.randh()
          params[which] = dnest4.wrap(params[which], 0.9, 1)
        elif which == grad_idx:
          params[which] += prior_vars[trap_idx]*np.int(dnest4.randh())
          params[which] = np.int(np.clip(params[which], 0, len(detector.gradList)-1))

        else: #velocity or rc params: cant be below 0, can be arb. large
          params[which] += prior_vars[which]*dnest4.randh()
          params[which] = dnest4.wrap(params[which], 0, 1E9)

        return logH

    def log_likelihood(self, params):
        """
        Gaussian sampling distribution.
        """
        b_over_a, c, d, rc1, rc2, rcfrac = params[tf_first_idx:tf_first_idx+6]
        h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0 = params[velo_first_idx:velo_first_idx+6]
        charge_trapping = params[trap_idx]
        grad = np.int(params[grad_idx])

        rad_arr, phi_arr, theta_arr, scale_arr, t0_arr, smooth_arr, m_arr, b_arr = params[len(priors):].reshape((8, num_waveforms))

        args = []
        # sum_like = 0
        for (wf_idx, wf) in enumerate(wfs):
            # print rad_arr[wf_idx]
            args.append([wf,  rad_arr[wf_idx], phi_arr[wf_idx], theta_arr[wf_idx],
                          scale_arr[wf_idx], t0_arr[wf_idx], smooth_arr[wf_idx],
                          m_arr[wf_idx], b_arr[wf_idx],
                         b_over_a, c, d, rc1, rc2, rcfrac,
                         h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0,
                         charge_trapping, grad
                        ])
            # sum_like += WaveformLogLike(wf,  rad_arr[wf_idx], phi_arr[wf_idx], theta_arr[wf_idx],
            #                scale_arr[wf_idx], t0_arr[wf_idx], smooth_arr[wf_idx],
            #                m_arr[wf_idx], b_arr[wf_idx],
            #                b_over_a, c, d, rc1, rc2, rcfrac,
            #                h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0,
            #                charge_trapping, grad
            #                )


        results = pool.map(WaveformLogLikeStar, args)

        sum_like = np.sum(results)

        return sum_like


def WaveformLogLikeStar(a_b):
  return WaveformLogLike(*a_b)

def WaveformLogLike(wf, rad, phi, theta, scale, t0, smooth, m, b, b_over_a, c, d, rc1, rc2, rcfrac, h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0, charge_trapping, grad):
    # print "theta is %f" % (theta/np.pi)
    r = rad#rad * np.cos(theta)
    z = theta#rad * np.sin(theta)

    if scale < 0 or t0 < 0:
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

    model = detector.MakeSimWaveform(r, phi, z, scale, t0, data_len, h_smoothing=smooth)
    if model is None:
      return -np.inf

    if np.amin(model) < 0:
      return -np.inf
    if model[-1] < 0.9*wf.wfMax:
      return -np.inf
    if np.argmax(model) == len(model)-1:
      return -np.inf

    #kill way too fast wfs
    t50_idx = findTimePointBeforeMax(model, 0.5)
    t50 = t50_idx - t0
    if t50 < 20 or t50 > 100:
        return -np.inf

    #kill way too slow wfs
    t50_max = np.argmax(model) - t50_idx


    if t50_max > 30:
        # print "killing ",
        # print np.argmax(model), t50
        return -np.inf


    baseline_trend = np.linspace(b, m*data_len+b, data_len)
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
