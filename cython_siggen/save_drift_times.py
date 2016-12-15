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
  det =  Detector(detName, temperature=tempGuess, timeStep=timeStepSize, numSteps=fitSamples*10)
  det.LoadFieldsGrad("fields_impgrad.npz", pcLen=pcLenGuess, pcRad=pcRadGuess)
  det.SetFieldsGradInterp(gradGuess)

  b_over_a = 0.107213
  c = -0.815152
  d = 0.822696
  rc1 = 74.4
  rc2 = 1.79
  rcfrac = 0.992
  trapping_rc = 120#us
  det.SetTransferFunction(b_over_a, c, d, rc1, rc2, rcfrac)
  #  det.trapping_rc = trapping_rc #us
  det.trapping_rc = trapping_rc
  
  det.siggenInst.set_velocity_type(1)
  h_100_mu0, h_100_beta, h_100_e0, h_111_mu0, h_111_beta, h_111_e0 = 66333., 0.744, 181., 107270., 0.580, 100.
  
  number = 100
  dt_matrix = np.empty((number,number)) * np.nan

  plt.figure()
  plt.xlim(0,75)

  for (r_idx, r) in enumerate(np.linspace(0, det.detector_radius, number)):
    for (z_idx, z) in enumerate(np.linspace(0, det.detector_length, number)):
      ml_wf = det.MakeSimWaveform(r, np.pi/8, z, 1, 0, fitSamples, h_smoothing=10)
      
      if ml_wf is not None:
        t50_idx  = findTimePointBeforeMax(ml_wf, 0.5)
        dt_matrix[r_idx, z_idx] = t50_idx

#        if z<0.5 and r< 15:
#          plt.plot(ml_wf, color="r")
        if z<0.5 and r> 15:
          plt.plot(ml_wf, color="b")
#  plt.figure()
#  plt.imshow(dt_matrix.T, origin='lower',  extent=(0, det.detector_radius, 0, det.detector_length), aspect='auto')
#  plt.colorbar()
#  plt.show()

  np.save("P42574A_drifttimes.npy", dt_matrix)


def findTimePointBeforeMax(data, percent):

  #don't screw up the data, bro
  int_data = np.copy(data)
  max_idx = np.argmax(int_data)
  int_data /= int_data[max_idx]
  
  int_data = int_data[0:max_idx]

  return np.where(np.less(int_data, percent))[0][-1]


#  plt.show()

if __name__=="__main__":
    test_velo()