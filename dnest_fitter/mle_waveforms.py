#!/usr/local/bin/python
import matplotlib
#matplotlib.use('CocoaAgg')
import sys, os, time
import scipy.optimize as op
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

import helpers
from pysiggen import Detector

from progressbar import ProgressBar, Percentage, Bar, ETA
from multiprocessing import Pool
from timeit import default_timer as timer

max_sample_idx = 125

#Prepare detector
timeStepSize = 1
fitSamples = 200
detName = "conf/P42574A_ben.conf"
detector =  Detector(detName, timeStep=timeStepSize, numSteps=fitSamples*10./timeStepSize, maxWfOutputLength=fitSamples + max_sample_idx + 2 )
fieldFileName = "P42574A_fields_impgrad_0.00000-0.00100.npz"
#sets the impurity gradient.  Don't bother changing this
detector.LoadFieldsGrad(fieldFileName)


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

detector.SetFieldsGradIdx(np.int(imp_grad))
detector.siggenInst.set_hole_params(h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0)
detector.trapping_rc = trapping_rc
detector.SetTransferFunction(tf_b, tf_c, tf_d, rc1, rc2, rcfrac, )

fallPercentage = 0.985

rad_mult = 10.
phi_mult = 0.1
z_mult = 10.
scale_mult = 1000.
maxt_mult = 100.
smooth_mult = 10.

out_file_name = "mle_wfs.txt"

def main(argv):

    # numThreads = 8

    wfFileName = "fep_event_set_runs11510-11539_channel626.npz"
    #wfFileName = "P42574A_24_spread.npz"
    if os.path.isfile(wfFileName):
        data = np.load(wfFileName)
        wfs = data['wfs']
        numWaveforms = wfs.size
    else:
        print "No saved waveforms available."
        exit(0)

    print "attempting to fit %d waveforms" % numWaveforms
    
    doPlot = 0
    if doPlot:
        plt.ion()
        fig1 = plt.figure(0, figsize=(15,7))
        plt.clf()
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1], sharex=ax0)
        ax1.set_xlabel("Digitizer Time [ns]")
        ax0.set_ylabel("Voltage [Arb.]")
        ax1.set_ylabel("Residual")

    # bar = ProgressBar(widgets=[Percentage(), Bar(), ETA()], maxval=len(wfs)).start()
    global waveform

    start = timer()
    with open(out_file_name, "w") as text_file:

        for (idx,wf) in enumerate(wfs):

            # if idx < 16: continue

            wf_idx = idx
            wf.WindowWaveformAroundMax(fallPercentage=fallPercentage, rmsMult=2, earlySamples=max_sample_idx)
            waveform = wf
            dataLen = wf.wfLength
            t_data = np.arange(dataLen) * 10

            # rad = np.sqrt(15**2+15**2)
            # theta = np.pi/4
            r,phi, z, scale, maxt, smooth  = 25,np.pi/8, 25, wf.wfMax, max_sample_idx, 10
            r /= rad_mult
            phi /= phi_mult
            z /= z_mult
            scale /= scale_mult
            maxt /= maxt_mult
            smooth /= smooth_mult

            minresult = None
            minlike = np.inf

            # rsteps = 4
            # rstepsize = detector.detector_radius / (rsteps+1)
            #
            # thetasteps = 4
            # zstepsize = np.pi/
            #
            # for rad in np.linspace(np.sqrt(2*10**2), np.sqrt(2*30**2), rsteps):
            #     for theta in np.linspace(np.pi/2/5, zsteps):
            #         # r /= rad_mult
            #         # z /= z_mult
            #         startGuess = [r, phi, theta, scale, maxt, smooth ]
            #         result = op.minimize(nll, startGuess,   method="Powell")
            #         if result['fun'] < minlike:
            #           minlike = result['fun']
            #           minresult = result

            bounds =[(0, detector.detector_radius/rad_mult), (0, np.pi/4/phi_mult), (0, detector.detector_length/z_mult),
                     (scale -50/scale_mult, scale+50/scale_mult), (maxt-5/maxt_mult, maxt+5/maxt_mult), (0,25./smooth_mult), ]


            result = op.differential_evolution(nll, bounds, polish=False,)# strategy='best1bin', mutation=(1, 1.5))

            # startGuess = [r,phi, z, scale, maxt, smooth, 1]
            # result = op.basinhopping(nll, startGuess, )

            # result = op.minimize(nll, startGuess,   method="Nelder-Mead", options={"maxfev": 10E4})

            r, phi, z, scale, maxt, smooth, = result["x"]

            # startGuess = r, phi, z, scale, maxt, smooth,
            # result = op.minimize(nll, startGuess,   method="Powell")
            # r, phi, z, scale, maxt, smooth, = result["x"]
            # r = rad * np.cos(theta)
            # z = rad * np.sin(theta)


            r *= rad_mult
            phi *= phi_mult
            z *= z_mult
            scale *= scale_mult
            maxt *= maxt_mult
            smooth *= smooth_mult

            print "wf %d best fit like %f" % (wf_idx,result["fun"])
            print " --> at ",
            print  r, phi, z, scale, maxt, smooth

            if doPlot:
                ax0.plot(t_data, wf.windowedWf, color="black")
                mle_wf = detector.MakeSimWaveform(r, phi, z, scale, maxt, dataLen, h_smoothing=smooth, alignPoint="max", doMaxInterp=False)
                # ax0.cla()
                # ax1.cla()
                ax0.plot(t_data, mle_wf, )
                ax1.plot(t_data, mle_wf - wf.windowedWf, )

                value = raw_input('  --> Press q to quit, any other key to continue\n')
                if value == 'q': exit(0)


            string = "%d  %d  %d %f  %f  %f  %f  %f  %f %f\n" % (wf.runNumber, wf.entry_number, len(wf.windowedWf), result["fun"], r, phi, z, scale, maxt, smooth )
            text_file.write(string)

    end = timer()
    print "total time: " + str(end-start)
    value = raw_input('  --> Press q to quit, any other key to continue\n')
    if value == 'q': exit(0)
        # bar.finish()
        # wfFileName += "_mlefit.npz"
        # np.savez(wfFileName, wfs = wfs )

def nll(*args):
  return -WaveformLogLike(*args)

def WaveformLogLike(theta):
    r, phi, z, scale, maxt, smooth,   = theta
    # d = dc * c
    # detector.SetTransferFunction(b, c, d, rc1, rc2, rcfrac, )

    # r,z = rad,theta
    # r = rad * np.cos(theta)
    # z = rad * np.sin(theta)

    r *= rad_mult
    phi *= phi_mult
    z *= z_mult
    scale *= scale_mult
    maxt *= maxt_mult
    smooth *= smooth_mult

    if scale < 0:
      return -np.inf
    if smooth < 0:
       return -np.inf
    if not detector.IsInDetector(r, phi, z):
      return -np.inf

    data = waveform.windowedWf
    model_err = waveform.baselineRMS
    data_len = len(data)

    model = detector.MakeSimWaveform(r, phi, z, scale, maxt, data_len, h_smoothing=smooth, alignPoint="max", doMaxInterp=False)
    if model is None:
        return -np.inf

    inv_sigma2 = 1.0/(model_err**2)
    ln_like = -0.5*(np.sum((data-model)**2*inv_sigma2 - np.log(inv_sigma2)))
    return ln_like

if __name__=="__main__":
    main(sys.argv[1:])
