import numpy as np
import sys, os, shutil
import dnest4

import numpy.random as rng
from multiprocessing import Pool
from pysiggen import Detector
import matplotlib.pyplot as plt
from matplotlib import gridspec


max_sample_idx = 125

rc1 = 73.085166
rc2 = 1.138420
rcfrac = 0.997114

h_100_mu0 = 5226508.435728
tf_phi = 1.527227
imp_grad = 0.000000
h_100_lnbeta = 1.657842
tf_omega = 0.134503
trapping_rc = 441.635318
h_100_emu = 57822415.222726
d = 0.815074
h_111_mu0 = 3433054.187637
rc1 = 72.671781
h_111_lnbeta = 0.854014
rc2 = 2.205759
h_111_emu = 7023863.173350
rcfrac = 0.996002

#convert velo params
h_100_beta = 1./np.exp(h_100_lnbeta)
h_111_beta = 1./np.exp(h_111_lnbeta)
h_100_e0 = h_100_emu / h_100_mu0
h_111_e0 = h_111_emu / h_111_mu0

#convert tf params
c = -d * np.cos(tf_omega)
b_ov_a = c - np.tan(tf_phi) * np.sqrt(d**2-c**2)
a = 1./(1+b_ov_a)
tf_b = a * b_ov_a
tf_c = c
tf_d = d

def main(argv):


    wfFileName = "P42574A_24_spread.npz"

    if os.path.isfile(wfFileName):
        data = np.load(wfFileName)
        wfs = data['wfs']
    else:
        print "No saved waveforms available."
        exit(0)

    global wf
    wf = wfs[16]

    max_sample_idx = 150
    fallPercentage = 0.99
    wf.WindowWaveformAroundMax(fallPercentage=fallPercentage, rmsMult=2, earlySamples=max_sample_idx)
    baseline_length = wf.t0Guess

    print "wf length %d (entry %d from run %d)" % ( wf.wfLength, wf.entry_number, wf.runNumber)
    wf_length = wf.wfLength
    wf_max= np.argmax(wf.windowedWf)
    baseline_origin_idx = baseline_length - 30

    if baseline_origin_idx < 0:
        print "not enough baseline!!"
        exit(0)
    initT0Padding(max_sample_idx, baseline_origin_idx)

    siggen_wf_length = (max_sample_idx - baseline_length + 10)*10

    global output_wf_length
    output_wf_length = wf_length + 1

    #setup .ector
    timeStepSize = 1
    fitSamples = 200
    detName = "conf/P42574A_ben.conf"
    det =  Detector(detName, timeStep=timeStepSize, numSteps=siggen_wf_length, maxWfOutputLength =output_wf_length, )
    fieldFileName = "P42574A_fields_impgrad_0.00000-0.00100.npz"
    #sets the impurity gradient.  Don't bother changing this
    det.LoadFieldsGrad(fieldFileName)

    det.SetFieldsGradIdx(np.int(imp_grad))
    det.siggenInst.set_hole_params(h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0)
    det.trapping_rc = trapping_rc
    det.SetTransferFunction(tf_b, tf_c, tf_d, rc1, rc2, rcfrac, )
    global detector
    detector = det



    directory = ""

    if len(argv) == 0:
        fit(directory)
    elif argv[0] == "plot":
        if len(argv) > 1: directory = argv[1]
        plot("sample.txt", directory)
    else:
        fit(argv[0])

def fit(directory):
    # Create a model object and a sampler
    model = Model()
    sampler = dnest4.DNest4Sampler(model,
                                 backend=dnest4.backends.CSVBackend(basedir ="./" + directory,
                                                                    sep=" "))

    # Set up the sampler. The first argument is max_num_levels
    gen = sampler.sample(max_num_levels=100, num_steps=10000, new_level_interval=10000,
                        num_per_step=1000, thread_steps=100,
                        num_particles=5, lam=10, beta=100, seed=1234)

    # Do the sampling (one iteration here = one particle save)
    for i, sample in enumerate(gen):
      print("# Saved {k} particles.".format(k=(i+1)))

def plot(sample_file_name, directory, plotNum=100):
    fig1 = plt.figure(0, figsize=(20,10))
    plt.clf()
    gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharex=ax0)
    ax1.set_xlabel("Digitizer Time [ns]")
    ax0.set_ylabel("Voltage [Arb.]")
    ax1.set_ylabel("Residual")


    dataLen = wf.wfLength
    t_data = np.arange(dataLen) * 10
    ax0.plot(t_data, wf.windowedWf, color="black")

    sample_file_name = directory + sample_file_name
    if sample_file_name == directory + "sample.txt":
      shutil.copy(directory+ "sample.txt", directory+"sample_plot.txt")
      sample_file_name = directory + "sample_plot.txt"

    data = np.loadtxt( sample_file_name)
    num_samples = len(data)
    print "found %d samples" % num_samples,
    print " , plotting %d" % plotNum

    if sample_file_name== (directory+"sample_plot.txt"):
        if num_samples > plotNum: num_samples = plotNum

    r_arr = np.empty(num_samples)
    z_arr = np.empty(num_samples)

    for (idx,params) in enumerate(data[-num_samples:]):
        r, phi, z, scale, t0, smooth = params
        r_arr[idx], z_arr[idx] = r, z

        ml_wf = detector.MakeSimWaveform(r, phi, z, scale, t0,  np.int(output_wf_length), h_smoothing = smooth, alignPoint="max")
        if ml_wf is None:
            continue

        dataLen = wf.wfLength
        t_data = np.arange(dataLen) * 10
        ax0.plot(t_data, ml_wf[:dataLen], color="g", alpha=0.1)
        ax1.plot(t_data, ml_wf[:dataLen] -  wf.windowedWf, color="g",alpha=0.1)

    ax0.set_ylim(-20, wf.wfMax*1.1)
    ax1.set_ylim(-20, 20)



    positionFig = plt.figure(3, figsize=(10,10))
    plt.clf()
    # colorbars = ["Reds","Blues", "Greens", "Purples", "Oranges", "Greys", "YlOrBr", "PuRd"]
    xedges = np.linspace(0, np.around(detector.detector_radius,1), np.around(detector.detector_radius,1)*10+1)
    yedges = np.linspace(0, np.around(detector.detector_length,1), np.around(detector.detector_length,1)*10+1)
    plt.hist2d(r_arr, z_arr,  bins=[ xedges,yedges  ],  cmap=plt.get_cmap("Greens"), cmin=0.1)
    plt.xlabel("r from Point Contact (mm)")
    plt.ylabel("z from Point Contact (mm)")
    plt.xlim(0, detector.detector_radius)
    plt.ylim(0, detector.detector_length)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.show()


def initMultiThreading(numThreads):
  global num_threads
  global pool
  num_threads = numThreads
  if num_threads > 1:
      pool = Pool(num_threads, initializer=initializeDetector, initargs=[detector.__getstate__()])

def initT0Padding(maxt_pad, linear_baseline_origin):
    global maxt_guess, min_maxt, max_maxt, baseline_origin_idx
    maxt_guess = maxt_pad
    max_maxt = maxt_guess + 10
    min_maxt = maxt_guess - 10
    baseline_origin_idx = linear_baseline_origin


def get_t0_params():
    return (t0_guess, min_t0, max_t0, )

def draw_position():
  r = rng.rand() * detector.detector_radius
  z = rng.rand() * detector.detector_length
  scale = np.amax(wf.windowedWf)
  t0 = None

  if not detector.IsInDetector(r, 0.1, z):
    return draw_position()
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

    def from_prior(self):
        """
        Unlike in C++, this must *return* a numpy array of parameters.
        """
        (r,z, scale, t0) = draw_position()
        smooth_guess = 10

        phi = rng.rand() * np.pi/4
        scale = 5*rng.randn() + scale - .005*scale
        t0 = 3*rng.randn() + maxt_guess
        smooth = np.clip(rng.randn() + smooth_guess, 0, 20)
        # m_arr =  0.0001*rng.randn() + 0.
        # b_arr =  0.001*rng.randn() + 0.

        return [r, phi, z, scale, t0, smooth]

    def perturb(self, params):
        """
        Unlike in C++, this takes a numpy array of parameters as input,
        and modifies it in-place. The return value is still logH.
        """
        logH = 0.0

        which = rng.randint(len(params))


        if which == 0:
            params[which] += (detector.detector_radius)*dnest4.randh()
            params[which] = dnest4.wrap(params[which] , 0, detector.detector_radius)
        elif which == 1:
            max_val = np.pi/4
            params[which] += np.pi/4*dnest4.randh()
            params[which] = dnest4.wrap(params[which], 0, max_val)
            if params[which] < 0 or params[which] > np.pi/4:
                print "wtf phi"
            #params[which] = np.clip(params[which], 0, max_val)
        elif which == 2:
            params[which] += (detector.detector_length)*dnest4.randh()
            params[which] = dnest4.wrap(params[which] , 0, detector.detector_length)

        elif which == 3: #scale
            min_scale = wf.wfMax - 0.01*wf.wfMax
            max_scale = wf.wfMax + 0.005*wf.wfMax
            params[which] += (max_scale-min_scale)*dnest4.randh()
            params[which] = dnest4.wrap(params[which], min_scale, max_scale)
        #   print "  adjusted scale to %f" %  ( params[which])

        elif which == 4: #t0
          params[which] += 1*dnest4.randh()
          params[which] = dnest4.wrap(params[which], min_maxt, max_maxt)
        elif which == 5: #smooth
          params[which] += 0.1*dnest4.randh()
          params[which] = dnest4.wrap(params[which], 0, 25)
            #   print "  adjusted smooth to %f" %  ( params[which])

            # elif which == 6: #wf baseline slope (m)
            #     logH -= -0.5*(params[which]/1E-4)**2
            #     params[which] += 1E-4*dnest4.randh()
            #     logH += -0.5*(params[which]/1E-4)**2
            # elif which == 7: #wf baseline incercept (b)
            #     logH -= -0.5*(params[which]/1E-2)**2
            #     params[which] += 1E-2*dnest4.randh()
            #     logH += -0.5*(params[which]/1E-2)**2

            #   params[which] += 0.01*dnest4.randh()
            #   params[which]=dnest4.wrap(params[which], -1, 1)
            #   print "  adjusted b to %f" %  ( params[which])

        else: #velocity or rc params: cant be below 0, can be arb. large
            print "which value %d not supported" % which
            exit(0)


        return logH

    def log_likelihood(self, params):
        """
        Gaussian sampling distribution.
        """
        r, phi, z, scale, t0, smooth = params
        return WaveformLogLike(wf, r, phi, z, scale, t0, smooth)


def WaveformLogLike(wf, r, phi, z, scale, maxt, smooth):

    if scale < 0:
      return  -np.inf
    if smooth < 0:
      return  -np.inf
    if not detector.IsInDetector(r, phi, z):
      return  -np.inf

    data = wf.windowedWf
    model_err = wf.baselineRMS
    data_len = len(data)

    model = detector.MakeSimWaveform(r, phi, z, scale, maxt, data_len, h_smoothing=smooth, alignPoint="max")
    if model is None:
      return  -np.inf
    if np.any(np.isnan(model)):  return  -np.inf

    # start_idx = -bl_origin_idx
    # end_idx = data_len - bl_origin_idx - 1
    # baseline_trend = np.linspace(m*start_idx+b, m*end_idx+b, data_len)
    # model += baseline_trend

    inv_sigma2 = 1.0/(model_err**2)
    ln_like = -0.5*(np.sum((data-model)**2*inv_sigma2 - np.log(inv_sigma2)))
    return  ln_like

if __name__=="__main__":
    main(sys.argv[1:])
