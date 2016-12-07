from detector_model import Detector
import numpy as np
import matplotlib.pyplot as plt
from scipy import  signal, interpolate, ndimage, optimize


def test_velo():
  fitSamples = 200
  timeStepSize = 1

  #Prepare detector
  tempGuess = 77
  gradGuess = 0.051005
  pcRadGuess = 2.499387
  pcLenGuess = 1.553464
  
  #Create a detector model
  detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.05,2.5, 1.65)
  det =  Detector(detName, temperature=tempGuess, timeStep=timeStepSize, numSteps=fitSamples*10./timeStepSize,)
  det.LoadFields("P42574A_fields_v3.npz")
  det.SetFields(pcRadGuess, pcLenGuess, gradGuess)
  
  q = 1
  
#  for i in range(100):
#    det.siggenInst.set_velocity_type(1)
#    ben_arr = np.copy(det.MakeRawSiggenWaveform(15, np.pi/4, 15, 1, ))
#    e_arr = np.copy(det.MakeRawSiggenWaveform(33, np.pi/8, 38, -1, ))


  det.siggenInst.set_velocity_type(1)
  ben_arr = np.copy(det.MakeRawSiggenWaveform(15, np.pi/4, 15, 1, ))
  ben_arr2 = np.copy(det.MakeRawSiggenWaveform(15, 0, 15, q, ))
  
  det.siggenInst.set_hole_params(61824., 0.942, 185., 61215., 0.662, 182.)
  bruy_arr = np.copy(det.MakeRawSiggenWaveform(15, np.pi/4, 15, 1, ))
  bruy_arr2 = np.copy(det.MakeRawSiggenWaveform(15, 0, 15, q, ))
##  ben_arr3 = np.copy(det.MakeRawSiggenWaveform(15, np.pi/2, 15, q, ))
##  ben_arr4 = np.copy(det.MakeRawSiggenWaveform(15, np.pi/10, 15, q, ))
##  ben_arr5 = np.copy(det.MakeRawSiggenWaveform(15, np.pi/4 - np.pi/10, 15, q, ))

  det.siggenInst.set_velocity_type(0)
  david_arr = np.copy(det.MakeRawSiggenWaveform(15, 0, 15, q, ))
  david_arr2 = np.copy(det.MakeRawSiggenWaveform(15, np.pi/4, 15, q, ))

  plt.figure()
  plt.plot(ben_arr,  "blue", ls="--", label="Reggiani <110>") #slow axis
  plt.plot(ben_arr2,  "blue",  label="Reggiani <100>") #fast axis?
  
  
  plt.plot(bruy_arr,  "green", ls="--", label="Bruyneel <110>") #slow axis
  plt.plot(bruy_arr2,  "green",  label="Bruyneel <100>") #fast axis?
  
#  plt.plot(ben_arr3,  "blue",  ls="-.") #fast axis?
#  plt.plot(ben_arr4,  "cyan",) #just off the fast
#  plt.plot(ben_arr5,  "black",) #just off the slow

  plt.plot(david_arr2, "red", ls="--", label="David <110>")
  plt.plot(david_arr, "red", label="David <100>")
  plt.legend(loc=2)
  
  plt.xlim(0, 350)


  plt.xlabel("Time [ns]")
  plt.ylabel("Amplitude [a.u.]")

  plt.show()

#  theta = np.linspace(0, np.pi, 100)
#  theta_recon = np.empty(100)
#  
#  fig1 = plt.figure(1, figsize=(10,10))
#  r = 15.
#  
#  for (idx,th) in enumerate(theta):
#    z = r * np.cos(th)
#    plt.scatter(idx,z)
#    
##    print "x: %f, y: %f" % (x,y)
#
#    theta_recon[idx] = np.arccos(z/r)
#    theta_recon[idx] /= np.pi
#  
#  fig2 = plt.figure(2)
#  plt.plot(theta_recon)

#    plt.figure(1)
#    plt.scatter(x,y)


#  plt.show()

if __name__=="__main__":
    test_velo()