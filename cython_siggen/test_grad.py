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


  fitSamples = 130
  timeStepSize = 1

  tempGuess = 79.071172
  gradGuess = 0.04
  pcRadGuess = 2.5
  pcLenGuess = 1.6

  #Create a detector model
  detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.05,2.5, 1.65)
  det =  Detector(detName, temperature=tempGuess, timeStep=timeStepSize, numSteps=fitSamples*10)

  print "loading fields"
  det.LoadFieldsGrad("fields_impgrad.npz", pcLen=pcLenGuess, pcRad=pcRadGuess)

  det.siggenInst.set_velocity_type(1)


  print "ready to roll"

  plt.figure()

  for (grad_idx, grad) in enumerate(det.gradList):
    print "grad_idx %d, grad %0.4f" % (grad_idx, grad)

    det.SetFieldsGradIdx(grad_idx)

    wf_arr = np.copy(det.MakeRawSiggenWaveform(15, 0, 15, 1, ))

    plt.plot(wf_arr)


  plt.show()


if __name__=="__main__":
    test_velo()
