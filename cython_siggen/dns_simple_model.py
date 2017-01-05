import numpy as np
import scipy.stats as stats
import scipy.optimize as op
import dnest4

import numpy.random as rng

def initializeDetector(det):
  global detector
  detector = det
  detector.SetTransferFunction(5.31003292334, -0.808557803157, 0.815966976844, 81.8681451166, 3.6629565274, 0.995895193187)
  det.SetFieldsGradInterp(0.01)

def initializeWaveform( wf_init, wf_guess_result):
  global wf
  wf = wf_init

  global wf_guess
  wf_guess = wf_guess_result


def initializeDetectorAndWaveform(det, wf_init):
  initializeWaveform(wf_init)
  initializeDetector(det)
min_t0 = 10
max_t0 = 55
t0_pad = 50
priors = np.empty(8)

#linear baseline slope and intercept...
priors[6] = 0
priors[7] = 0

prior_vars =  np.empty(len(priors))
prior_vars[6:8] = 0.001, 0.01

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

        t0 = np.clip(0.5*rng.randn() + t0_pad, min_t0, max_t0)
        scale = 10*rng.randn() + wf_guess[3]
        smooth = np.clip(rng.randn() + wf_guess[5], 0, 20)

        m =  prior_vars[6]*rng.randn() + priors[6]
        b =  prior_vars[7]*rng.randn() + priors[7]

        print "\n"
        print "new waveform:"
        print "  wf params: ",
        print  r, phi, z, scale, t0, smooth, m, b

        return np.array([
              rad, phi, theta, scale, t0, smooth, m, b,
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
          params[4] = dnest4.wrap(params[4] + t0_jump , min_t0, max_t0)

          params[0] = np.clip(params[0] + r_jump , 0, max_rad)
          params[4] = np.clip(params[4] + t0_jump , min_t0, max_t0)

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
            dnest4.wrap(params[which], -0.01, 0.01)
            params[which] = np.clip(params[which], -0.01, 0.01)
          if which == 7:
            dnest4.wrap(params[which], -01, 01)
            params[which] = np.clip(params[which], -01, 01)
        else:
            print "which value %d not supported" % which
            exit(0)

        return logH

    def log_likelihood(self, params):
        """
        Gaussian sampling distribution.
        """
        rad, phi, theta, scale, t0, smooth = params[:6]
        m, b = params[6:8]

        r = rad * np.cos(theta)
        z = rad * np.sin(theta)

        if scale < 0 or t0 < 0:
          return -np.inf
        if smooth < 0:
           return -np.inf
        if not detector.IsInDetector(r, phi, z):
          return -np.inf

        data = wf.windowedWf
        model_err = wf.baselineRMS
        data_len = len(data)

        model = detector.MakeSimWaveform(r, phi, z, scale, t0, data_len, h_smoothing=smooth)
        if model is None:
          return -np.inf
        if np.any(np.isnan(model)):
          return -np.inf

        baseline_trend = np.linspace(b, m*data_len+b, data_len)
        model += baseline_trend

        inv_sigma2 = 1.0/(model_err**2)

        return -0.5*(np.sum((data-model)**2*inv_sigma2 - np.log(inv_sigma2)))
