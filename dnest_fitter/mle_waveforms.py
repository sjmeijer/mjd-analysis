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

#Create a detector model
timeStepSize = 1 #ns
detName = "conf/P42574A_bull.conf"
field_file_name = "P42574A_mar28_21by21.npz"
det =  Detector(detName, timeStep=timeStepSize, numSteps=fitSamples, maxWfOutputLength =fitSamples, t0_padding=100 )
det.LoadFieldsGrad(field_file_name)

#fit params
fallPercentage = 0.985
out_file_name = "mle_wfs.txt"

rad_mult = 10.
phi_mult = 0.1
z_mult = 10.
scale_mult = 1000.
maxt_mult = 100.
smooth_mult = 10.


#set up the detector w/ trained params
h_111_va = 6156919.714510
tf_phi = -1.516434
imp_grad = 0.072153
h_111_vmax = 8876474.471688
tf_omega = 0.134181
imp_avg = -0.345865
h_100_multa = 1.082742
tf_d = 0.815624
trapping_rc = 227.036549
h_100_multmax = 1.112970
rc1 = 73.066335
alias_rc = 9.930959
h_100_beta = 0.453146
rc2 = 1.262588
h_111_beta = 0.666700
rcfrac = 0.996449

c = -d * np.cos(tf_omega)
b_ov_a = c - np.tan(tf_phi) * np.sqrt(d**2-c**2)
a = 1./(1+b_ov_a)
tf_b = a * b_ov_a
tf_c = c
tf_d = d

h_100_va = h_100_multa * h_111_va
h_100_vmax = h_100_multmax * h_111_vmax

h_100_mu0, h_100_beta, h_100_e0 = get_velo_params(h_100_va, h_100_vmax, h_100_beta)
h_111_mu0, h_111_beta, h_111_e0 = get_velo_params(h_111_va, h_111_vmax, h_111_beta)

det.SetTransferFunction(tf_b, tf_c, tf_d, rc1, rc2, rcfrac)
det.siggenInst.set_hole_params(h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0)
det.trapping_rc = charge_trapping
det.SetGrads(grad, imp_avg)
det.SetAntialiasingRC(aliasrc)

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

    doPlot = 1
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

            rsteps = 10
            rstepsize = detector.detector_radius / (rsteps+1)

            zsteps = 10
            zstepsize = detector.detector_length / (zsteps+1)

            for r in np.linspace(rstepsize, detector.detector_radius - rstepsize, rsteps):
                for z in np.linspace(zstepsize, detector.detector_length - zstepsize, zsteps):
                    r /= rad_mult
                    z /= z_mult
                    startGuess = [r, phi, z, scale, maxt, smooth ]
                    result = op.minimize(nll, startGuess,   method="Powell")
                    if result['fun'] < minlike:
                      minlike = result['fun']
                      minresult = result


            result = minresult

            # bounds =[(0, detector.detector_radius/rad_mult), (0, np.pi/4/phi_mult), (0, detector.detector_length/z_mult),
            #          (scale -50/scale_mult, scale+50/scale_mult), (maxt-5/maxt_mult, maxt+5/maxt_mult), (0,25./smooth_mult), ]
            #
            #
            # result = op.differential_evolution(nll, bounds, polish=False,)# strategy='best1bin', mutation=(1, 1.5))

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


            string = "%d  %d  %d  %f  %f  %f  %f  %f  %f  %f\n" % (wf.runNumber, wf.entry_number, len(wf.windowedWf), result["fun"], r, phi, z, scale, maxt, smooth )
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

def get_velo_params( v_a, v_max, beta):
    E_a = 500
    E_0 = np.power( (v_max*E_a/v_a)**beta - E_a**beta , 1./beta)
    mu_0 = v_max / E_0

    return (mu_0,  beta, E_0)
if __name__=="__main__":
    main(sys.argv[1:])
