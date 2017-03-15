from pysiggen import Detector
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def plot_waveforms():

    #the pysiggen Detector object is a wrapper around siggen that also does all the elctronics processing.  Let's set one up.

    wf_length = 350 #in tens of ns.  This sets how long a wf you will simulate (after post-processing for electronics etc)
    fitSamples = 800 #number of samples siggen will calculate (before post-processing, in 1 ns increments)
    timeStepSize = 1 #don't change this, you'll break everything
    detName = "conf/P42574A_ben.conf"
    detector =  Detector(detName, timeStep=timeStepSize, numSteps=fitSamples, maxWfOutputLength=5000)

    #Load field information
    detector.LoadFieldsGrad("P42574A_fields_impAndAvg_21by21.npz")

    #Set a specific impurity gradient, impurity level combo
    imp_grad =detector.gradList[3];
    avg_imp = detector.impAvgList[3];
    detector.SetGrads(imp_grad, avg_imp)

    #hole drift mobilities
    detector.siggenInst.set_hole_params(66333., 0.744, 181., 107270., 0.580, 100.,)

    #Electronics transfer function
    rc_decay = 72.6 #us
    detector.SetTransferFunction(1, -0.814072377576, 0.82162729751, rc_decay, 1, 1)

    #position we're gonna simulate
    r, phi,z = 30, np.pi/8 ,30
    #energy just scales the simulated waveform amplitude linearly
    energy = 10
    #align time is where the signal is aligned.  If alignPoint="max" is set, this is at the signal max.  Else, its at the signal t0 (with flat baseline before t0)
    align_t = 200
    smooth = 10 #sigma of the gaussian charge size convolution

    #Set up some plotz
    fig1 = plt.figure(0, figsize=(15,8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[4, 1])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[2], sharex=ax0)
    ax1.set_xlabel("Digitizer Time [ns]")
    ax0.set_ylabel("Voltage [A.U.]")
    ax1.set_ylabel("Residual [A.U]")
    ax0.set_title("Raw hole signal (no processing)")

    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[3], sharex=ax2)
    ax3.set_xlabel("Digitizer Time [ns]")
    ax2.set_ylabel("Voltage [A.U.]")
    ax3.set_ylabel("Residual [A.U]")
    ax2.set_title("After electronics processing")

    timesteps_raw = np.arange(0, fitSamples) #for raw signal
    timesteps = np.arange(0, wf_length)*10. #for processed signal

    #need to do a np.copy because the memory allocated for wf sim gets rewritten with each sim
    wf_raw = np.copy(detector.MakeRawSiggenWaveform(r, phi, z,1)) #the 1 is to do holes - a -1 would do electrons
    wf_proc = np.copy(detector.MakeSimWaveform(r, phi, z, energy, align_t, wf_length, h_smoothing=smooth, trapType="fullSignal", alignPoint="max"))

    ax0.plot(timesteps_raw, wf_raw,  color="green")
    ax2.plot(timesteps, wf_proc,  color="green")

    #take a look at how changing average impurity level changes the signal
    num = 7
    imp_vals = np.linspace(detector.impAvgList[0], detector.impAvgList[-1], num)
    imp_grads = np.linspace(detector.gradList[0], detector.gradList[-1], num)

    #just to make sure the residual color is the same as the corresponding wf
    colorList = ["red", "blue", "purple", "orange", "black", "brown", "magenta", "cyan"]

    # for (idx, grad) in enumerate(imp_grads):
    for (idx, avg_imp) in enumerate(imp_vals):
        detector.SetGrads(imp_grad, avg_imp)

        new_wf_raw = np.copy(detector.MakeRawSiggenWaveform(r, phi, z, 1))
        ax0.plot(timesteps_raw, new_wf_raw,  color=colorList[idx])
        ax1.plot(timesteps_raw, wf_raw - new_wf_raw, color=colorList[idx])

        new_wf_proc= np.copy(detector.MakeSimWaveform(r, phi, z, energy, align_t, wf_length, h_smoothing=smooth, trapType="fullSignal", alignPoint="max"))
        ax2.plot(timesteps, new_wf_proc,  color=colorList[idx])
        ax3.plot(timesteps, wf_proc - new_wf_proc, color=colorList[idx])

    plt.show()


if __name__=="__main__":
    plot_waveforms()
