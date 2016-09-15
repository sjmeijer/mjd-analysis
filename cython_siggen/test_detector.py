from detector_model import Detector
import numpy as np
import matplotlib.pyplot as plt
from scipy import  signal, interpolate, ndimage

fitSamples = 100

#Prepare detector
zero_1 = -5.56351644e+07
pole_1 = -1.38796386e+04
pole_real = -2.02559385e+07
pole_imag = 9885315.37450211

zeros = [zero_1,0 ]
poles = [ pole_real+pole_imag*1j, pole_real-pole_imag*1j, pole_1]
system = signal.lti(zeros, poles, 1E7 )

tempGuess = 77.89
gradGuess = 0.0483
pcRadGuess = 2.591182
pcLenGuess = 1.613357

#Create a detector model
detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.05,2.5, 1.65)
det =  Detector(detName, temperature=tempGuess, timeStep=10., numSteps=fitSamples, tfSystem=system)
det.LoadFields("P42574A_fields_v3.npz")
det.SetFields(pcRadGuess, pcLenGuess, gradGuess)

print "time steps out number is %d" % det.siggenInst.GetSafeConfiguration()['ntsteps_out']
print "time steps calc number is %d" % det.siggenInst.GetSafeConfiguration()['time_steps_calc']
print "step size out is %f" % det.siggenInst.GetSafeConfiguration()['step_time_out']
print "step size calc is %f" % det.siggenInst.GetSafeConfiguration()['step_time_calc']

#sig_array = det.GetRawSiggenWaveform( 0.174070, 0.528466, 10.755795)
sig_array = det.GetSimWaveform( 0.174070, 0.528466, 3.755795, 1., 10, fitSamples)

plt.figure()
plt.plot(sig_array)
plt.ylim(-0.05, 1.05)
plt.xlabel("Time [ns]")
plt.ylabel("Charge [arb.]")
plt.show()