import numpy as np
import scipy.stats as stats
import scipy.optimize as op
import dnest4

import numpy.random as rng

def initializeDetector(det):
  global detector
  detector = det

def initializeWaveform( wf_init, wf_guess_result):
  global wf
  wf = wf_init

  global wf_guess
  wf_guess = wf_guess_result


def initializeDetectorAndWaveform(det, wf_init):
  initializeWaveform(wf_init)
  initializeDetector(det)


tf_first_idx = 8
velo_first_idx = 14
trap_idx = 20
grad_idx = 21
max_t0 = 105

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

priors = np.empty(22)

#linear baseline slope and intercept...
priors[6] = 0
priors[7] = 0

priors[tf_first_idx:tf_first_idx+3] = ba_prior, c_prior, d_prior
priors[rc1_idx:rc1_idx+3] = rc1_prior, rc2_prior, rc_frac_prior
priors[velo_first_idx:velo_first_idx+3] = h_100_mu0_prior, h_100_beta_prior, h_100_e0_prior
priors[velo_first_idx+3:velo_first_idx+6] = h_111_mu0_prior, h_111_beta_prior, h_111_e0_prior

priors[trap_idx] = 120.

prior_vars =  np.empty(len(priors))

prior_vars[6:8] = 0.001, 0.01

prior_vars[rc1_idx:rc1_idx+3] = 0.05*rc1_prior, 0.05*rc2_prior, 0.001

var = 0.01
prior_vars[velo_first_idx:velo_first_idx+6] = var*priors[velo_first_idx:velo_first_idx+6]
prior_vars[trap_idx] = 1.

priors[grad_idx] = 6
prior_vars[grad_idx] = 1


def draw_position():
#  det_max = np.sqrt(detector.detector_radius**2 + detector.detector_length**2)

  # r = rng.rand() * detector.detector_radius
  # z = rng.rand() * detector.detector_radius

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

  r, phi, z = wf_guess[0:3]
  r += rng.randn()*0.1
  z += rng.randn()*0.1
  if not detector.IsInDetector(r, 0.1, z):
#    print "not in detector..."
    return draw_position()
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
        pass



    def from_prior(self):
        """
        Unlike in C++, this must *return* a numpy array of parameters.
        """
        # 6 waveform parameters
#        print "location: (%f, %f)" % (r, z)

        (r,z) = draw_position()
        rad = np.sqrt(r**2+z**2)
        theta = np.tan(z/r)
        phi = rng.rand() * np.pi/4


#        det_max = np.sqrt(detector.detector_radius**2 + detector.detector_length**2)
#        rad = rng.rand() * det_max
#        phi = rng.rand() * np.pi/4
#        z = rng.rand() * np.pi/2

        t0 = np.clip(0.5*rng.randn() + 100, 0, max_t0)
        scale = 10*rng.randn() + wf_guess[3]
        smooth = np.clip(rng.randn() + wf_guess[5], 0, 20)

        m =  prior_vars[6]*rng.randn() + priors[6]
        b =  prior_vars[7]*rng.randn() + priors[7]

#        import matplotlib.pyplot as plt
#        plt.figure()
#
#        for idx in range(len(location_idxs[0])):
#          r_idx= location_idxs[0][idx]
#          z_idx= location_idxs[1][idx]
#          plt.scatter(r_arr[r_idx], z_arr[z_idx])
#
#          print "location: (%f, %f)" % (r_arr[r_idx], z_arr[z_idx])
#        plt.xlim(0, detector.detector_radius)
#        plt.ylim(0, detector.detector_length)
#        plt.show()
#
#        print "wf max is %f, scale set to %f" % (wf.wfMax, scale)

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

        print "\n"
        print "new waveform:"
        print "  wf params: ",
        print  r, phi, z, scale, t0, smooth, m, b
        print "  tf params: ",
        print b_over_a, c, d, rc1, rc2, rcfrac
        print "  velo params: ",
        print h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0


#        import matplotlib.pyplot as plt
#        from matplotlib import gridspec
#        plt.ion()
#        fig1 = plt.figure(1, figsize=(20,10))
#        plt.clf()
#        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
#        ax0 = plt.subplot(gs[0])
#        ax1 = plt.subplot(gs[1], sharex=ax0)
#        ax1.set_xlabel("Digitizer Time [ns]")
#        ax0.set_ylabel("Voltage [Arb.]")
#        ax1.set_ylabel("Residual")
#
#        dataLen = wf.wfLength
#        t_data = np.arange(dataLen) * 10
#        ax0.plot(t_data, wf.windowedWf, color="r")
#
#        fitSamples = 300
#        detector.SetTransferFunction(b_over_a, c, d, rc1, rc2, rcfrac)
#        detector.siggenInst.set_hole_params(h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0)
#        ml_wf = detector.MakeSimWaveform(r, phi, z, scale, t0,  fitSamples, h_smoothing = smooth)
#        ax0.plot(t_data, ml_wf[:dataLen], color="g", alpha=0.5)
#        ax1.plot(t_data, ml_wf[:dataLen] -  wf.windowedWf, color="g",alpha=0.5)
#
#
#        value = raw_input('  --> Press q to quit, s to skip, any other key to continue\n')
#        if value == 'q':
#          exit(0)


        return np.array([
              r, phi, theta, scale, t0, smooth, m, b,
              b_over_a, c, d,
              rc1, rc2, rcfrac,
              h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0,
              charge_trapping, grad
            ])

    def perturb(self, params):
        """
        Unlike in C++, this takes a numpy array of parameters as input,
        and modifies it in-place. The return value is still logH.
        """
        logH = 0.0
        which = rng.randint(len(params))

        if which == 0 or which == 4: #radius and t0
          max_rad = np.sqrt(detector.detector_radius**2 + detector.detector_length**2)

          mean = [0, 0]
          cov = [[1, -0.8], [-0.8, 1]]
          jumps = np.array((0.1*dnest4.randh(), 0.1*dnest4.randh()))
          (r_jump, t0_jump) = np.dot(cov, jumps)

          params[0] = dnest4.wrap(params[0] + r_jump , 0, max_rad)
          params[4] = dnest4.wrap(params[4] + t0_jump , 0, max_t0)

          params[0] = np.clip(params[0] + r_jump , 0, max_rad)
          params[4] = np.clip(params[4] + t0_jump , 0, max_t0)

        elif which == 1 or which ==2: #phi or theta
          if which == 1: max_val = np.pi/4
          if which ==2: max_val = np.pi/2
          params[which] += dnest4.randh()
          params[which] = dnest4.wrap(params[which], 0, max_val)
          params[which] = np.clip(params[which], 0, max_val)

        elif which == 3: #scale
          params[which] += dnest4.randh()
          params[which] = dnest4.wrap(params[which], wf.wfMax - 10*wf.baselineRMS, wf.wfMax + 10*wf.baselineRMS)

#        elif which == 4: #t0
#          params[which] += 0.1*dnest4.randh()
#          params[which] = np.clip(params[which], 0, wf.wfLength)
        elif which == 5: #smooth
          params[which] += 0.1*dnest4.randh()
          params[which] = np.clip(params[which], 0, 15)
          max_val = np.inf

        elif which == 6 or which == 7: #m and b, respectively
          #normally distributed, no cutoffs
          params[which] += prior_vars[which]*dnest4.randh()
          if which == 6:
            dnest4.wrap(params[which], -0.02, 0.02)
            params[which] = np.clip(params[which], -0.05, 0.05)
          if which == 7:
            dnest4.wrap(params[which], -2, 2)
            params[which] = np.clip(params[which], -5, 5)

        elif which == ba_idx: #b over a
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
        rad, phi, theta, scale, t0, smooth = params[:6]
        m, b = params[6:8]
        b_over_a, c, d, rc1, rc2, rcfrac = params[tf_first_idx:tf_first_idx+6]

        r = rad * np.cos(theta)
        z = rad * np.sin(theta)

#        print h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0
#        print "scale is %f" % scale

#        rc1 = 80.013468
#        rc2 = 2.078342
#        rcfrac = 0.992

        if scale < 0 or t0 < 0:
          return -np.inf
        if smooth < 0:
           return -np.inf
        if not detector.IsInDetector(r, phi, z):
          return -np.inf

        detector.SetTransferFunction(b_over_a, c, d, rc1, rc2, rcfrac)

        h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0 = params[velo_first_idx:velo_first_idx+6]
        detector.siggenInst.set_hole_params(h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0)

        charge_trapping = params[trap_idx]
        detector.trapping_rc = charge_trapping

        grad = np.int(params[grad_idx])
        detector.SetFieldsGradIdx(grad)

        data = wf.windowedWf
        model_err = wf.baselineRMS
        data_len = len(data)

        model = detector.MakeSimWaveform(r, phi, z, scale, t0, data_len, h_smoothing=smooth)
        if model is None:
          return -np.inf
        if np.any(np.isnan(model)):
          return -np.inf

        if np.amin(model) < 0:
          return -np.inf

        baseline_trend = np.linspace(b, m*data_len+b, data_len)
        model += baseline_trend

        #make sure the last point is near where it should be
        if model[-1] < 0.9*wf.wfMax or model[-1] > wf.wfMax:
          return -np.inf
        if np.argmax(model) == len(model)-1:
          return -np.inf

        #kill way too fast wfs (from t0-t50)
        t50_idx = findTimePointBeforeMax(model, 0.5)
        t50 = t50_idx - t0
        if t50 < 20 or t50 > 100:
          return -np.inf

        #kill way too slow wfs (from t50-t100)
        t50_max = np.argmax(model) - t50_idx
        if t50_max > 30:
          return -np.inf


        inv_sigma2 = 1.0/(model_err**2)
        return -0.5*(np.sum((data-model)**2*inv_sigma2 - np.log(inv_sigma2)))


def findTimePointBeforeMax(data, percent):

  #don't screw up the data, bro
  int_data = np.copy(data)
  max_idx = np.argmax(int_data)
  int_data /= int_data[max_idx]

  int_data = int_data[0:max_idx]
  try:
      return np.where(np.less(int_data, percent))[0][-1]
  except IndexError:
      print data
      import matplotlib.pyplot as plt
      plt.figure()
      plt.plot(data)
      plt.show()
      exit(0)
