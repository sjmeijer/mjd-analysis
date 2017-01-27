import numpy as np
import scipy.stats as stats
import scipy.optimize as op
import dnest4
from multiprocessing import Pool

import numpy.random as rng

def initializeDetector(det):
  global detector
  detector = det

def initializeWaveforms( wfs_init, ):
  global wfs
  wfs = wfs_init
  global num_waveforms
  num_waveforms = len(wfs)

def initializeDetectorAndWaveforms(det, wf_init):
  initializeWaveforms(wf_init)
  initializeDetector(det)

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

tf_first_idx = 0
priors = np.empty(6) #6 + 2)

ba_idx, c_idx, dc_idx = np.arange(3)+ tf_first_idx
rc1_idx, rc2_idx, rcfrac_idx = np.arange(3)+ tf_first_idx+3

rc1_prior =  73.085166
rc2_prior = 1.138420
rc_frac_prior = 0.997114
priors[rc1_idx:rc1_idx+3] = rc1_prior, rc2_prior, rc_frac_prior

prior_vars =  np.empty(len(priors))
prior_vars[rc1_idx:rc1_idx+3] = 0.2, 0.3, 0.001

def draw_position(wf_idx):
   r = rng.rand() * detector.detector_radius
   z = rng.rand() * detector.detector_radius
   scale = np.amax(wfs[wf_idx].windowedWf)
   t0 = None
   if not detector.IsInDetector(r, 0.1, z):
 #    print "not in detector..."
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
             scale_arr[wf_idx] = 5*rng.randn() + scale
             t0_arr[wf_idx] = 3*rng.randn() + maxt_guess
             smooth_arr[wf_idx] = np.clip(rng.randn() + smooth_guess, 0, 20)
             m_arr[wf_idx] =  0.0001*rng.randn() + 0.
             b_arr[wf_idx] =  0.001*rng.randn() + 0.

         b = 70*rng.rand() - 50

         log_gain = 10*rng.rand() - 10
         gain = np.exp(log_gain)

         d2 =  0.25 *rng.rand() + 0.5

         #limit from 60 to 90
         rc1 = dnest4.wrap(prior_vars[rc1_idx]*rng.randn() + priors[rc1_idx], 65, 80)
         rc2 = dnest4.wrap(prior_vars[rc2_idx]*rng.randn() + priors[rc2_idx], 0.1, 10)
         rcfrac = dnest4.wrap(prior_vars[rcfrac_idx]*rng.randn() + priors[rcfrac_idx], 0.9, 1)

         return np.hstack([
               b, gain, d2,
               rc1, rc2, rcfrac,
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
               params[which] += dnest4.randh()
               params[which] = dnest4.wrap(params[which], wf.wfMax - 10*wf.baselineRMS, wf.wfMax + 10*wf.baselineRMS)
               params[which] = np.clip(params[which], wf.wfMax - 50*wf.baselineRMS, wf.wfMax + 50*wf.baselineRMS)
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
           params[which] += 70*dnest4.randh()
           params[which] = dnest4.wrap(params[which], -50, 20)
         elif which == c_idx: #b over a
            #this is now 1/gain (so assume it goes from 1 to e^-...7?)
            log_gain = np.log(params[which])
            log_gain += 10*dnest4.randh()
            log_gain = dnest4.wrap(log_gain, -10, 0.0)
            params[which] = np.exp(log_gain)
            #  params[which] += 4*dnest4.randh()
            #  params[which] = dnest4.wrap(params[which], -2, 2)
         elif which == dc_idx: #b over a
             params[which] += 1*dnest4.randh()
             params[which] = dnest4.wrap(params[which], 0, 1)

         elif which == rc1_idx or which == rc2_idx or which == rcfrac_idx:
             #all normally distributed priors
             logH -= -0.5*((params[which] - priors[which])/prior_vars[which])**2
             params[which] += prior_vars[which]*dnest4.randh()
             logH += -0.5*((params[which] - priors[which])/prior_vars[which])**2

         else: #velocity or rc params: cant be below 0, can be arb. large
             print "which value %d not supported" % which
             exit(0)


         return logH

     def log_likelihood(self, params):
         """
         Gaussian sampling distribution.
         """
         tf_b, tf_c, tf_dc, rc1, rc2, rcfrac = params[tf_first_idx:tf_first_idx+6]
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
                               tf_b, tf_c, tf_dc, rc1, rc2, rcfrac,
                               baseline_origin_idx
                             ])

             results = pool.map(WaveformLogLikeStar, args)
             sum_like = np.sum(results)
         else:
             for (wf_idx, wf) in enumerate(wfs):
                 sum_like += WaveformLogLike(wf,  rad_arr[wf_idx], phi_arr[wf_idx], theta_arr[wf_idx],
                               scale_arr[wf_idx], t0_arr[wf_idx], smooth_arr[wf_idx],
                               m_arr[wf_idx], b_arr[wf_idx],
                               tf_b, tf_c, tf_dc, rc1, rc2, rcfrac,
                               baseline_origin_idx
                             )

         return sum_like


def WaveformLogLikeStar(a_b):
   return WaveformLogLike(*a_b)

def WaveformLogLike(wf, r, phi, z, scale, maxt, smooth, m, b, tf_b, tf_gain, tf_d2, rc1, rc2, rcfrac,bl_origin_idx):
     # #TODO: This needs to be length normalized somehow

     if scale < 0:
       return -np.inf
     if smooth < 0:
        return -np.inf
     if not detector.IsInDetector(r, phi, z):
       return -np.inf

     tf_2c = tf_gain - 1 - tf_d2

     detector.SetTransferFunction(tf_b, tf_2c, tf_d2, rc1, rc2, rcfrac, isDirect=True)

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
