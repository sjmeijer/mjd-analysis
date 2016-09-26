from detector_model import Detector
import numpy as np
import matplotlib.pyplot as plt
from scipy import  signal, interpolate, ndimage


def test_detector():
  fitSamples = 100

#  Prepare detector
  zero_1 = -5.56351644e+07
  pole_1 = -1.38796386e+04
  pole_real = -2.02559385e+07
  pole_imag = 9885315.37450211
  
  gain = 2./1e-8
  
  zeros = [zero_1,0. ]
  poles = [ pole_real+pole_imag*1j, pole_real-pole_imag*1j, pole_1]
  system = signal.lti(zeros, poles, gain )
  
  system.to_tf()
  num = system.num
  den = system.den
  
  print "original num: " + str(num)
  print "original den" + str(den)
  
  system.to_zpk()
  print "back to the zpk"
  print system.zeros
  print system.poles
  print system.gain

#  num = np.array((3478247474.8078203, 1.9351287044375424e+17, 6066014749714584.0)) 
#  den = [1, 40525756.715025946, 508584795912802.44, 7.0511687850000589e+18]
#  system = signal.lti(num, den)
#
#  zeros = system.zeros
#  poles = system.poles
#  gain = system.gain

  new_num, new_den, dt = signal.cont2discrete((num, den), 1E-8, method="bilinear")
#  new_num /= 1.05836038e-08
  print "new discrete tf representation"
  print new_num
  print new_den
  print dt

  new_z, new_p, new_k, dt = signal.cont2discrete((zeros, poles, gain), 1E-8, method="bilinear" )
  print "new discrete zpk representation"
  print new_z
  print new_p
  print new_k
  print dt
  
  print "...and back to tf"
  dis_num, dis_den = signal.zpk2tf(new_z, new_p, new_k)
  print dis_num
  print dis_den

  tempGuess = 77.89
  gradGuess = 0.0483
  pcRadGuess = 2.591182
  pcLenGuess = 1.613357

  #Create a detector model
  detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.05,2.5, 1.65)
  det =  Detector(detName, temperature=tempGuess, timeStep=10., numSteps=fitSamples, )
  det.LoadFields("P42574A_fields_v3.npz")
  det.SetFields(pcRadGuess, pcLenGuess, gradGuess)

  print "time steps out number is %d" % det.siggenInst.GetSafeConfiguration()['ntsteps_out']
  print "time steps calc number is %d" % det.siggenInst.GetSafeConfiguration()['time_steps_calc']
  print "step size out is %f" % det.siggenInst.GetSafeConfiguration()['step_time_out']
  print "step size calc is %f" % det.siggenInst.GetSafeConfiguration()['step_time_calc']


  sig_array = det.GetRawSiggenWaveform( 0.174070, 0.528466, 10.755795)
  
  t, sig_array2, x = signal.lsim(system, sig_array, det.time_steps, interp=False)
  
  
  sig_array1 = signal.lfilter(new_num[0], new_den, sig_array)
  sig_array4 = signal.lfilter(dis_num, dis_den, sig_array)
  
#  dissystem = system.to_discrete(1E-8)
#  
##  new_k = dissystem.zeros
##  new_p = dissystem.poles
##  new_k = dissystem.gain
##  dis_num, dis_den = signal.zpk2tf(new_z, new_p, new_k)
##  sig_array5 = signal.lfilter(dis_num, dis_den, sig_array)


#
#
  plt.figure()
  plt.plot(sig_array1, "b:")
  plt.plot(sig_array2, color="black")
  plt.plot(sig_array4, "r:")
#  plt.plot(sig_array5, "g:")

  plt.figure()
  plt.plot(sig_array2-sig_array1, color="b")
  plt.plot(sig_array2-sig_array4, color="r")
#  plt.plot(sig_array2-sig_array5, color="g")

  #plt.ylim(-0.05, 1.05)
  plt.xlabel("Time [ns]")
  plt.ylabel("Charge [arb.]")
  plt.show()

#  system.to_ss()
#  for i in range(10000):
#    sig_array = det.GetRawSiggenWaveform( 0.174070, 0.528466, 10.755795)
#    sig_array_out = signal.lfilter(new_num[0], new_den, sig_array)
    #sig_array_out = signal.lsim(system, sig_array, det.time_steps, interp=False)
  #  sig_array = det.GetSimWaveform( 0.174070, 0.528466, 3.755795, 1., 10, fitSamples)

if __name__=="__main__":
    test_detector()