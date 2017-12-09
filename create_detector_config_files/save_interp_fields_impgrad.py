#!/usr/local/bin/python
from ROOT import *

import matplotlib
import sys, os
import matplotlib.pyplot as plt

import numpy as np
from scipy import signal

from detector_model import *

def main(argv):


  fitSamples = 250 #has to be longer than the longest wf you're gonna fit
  tempGuess = 81.5
  
  #Set up detectors
  num = [3.64e+09, 1.88e+17, 6.05e+15]
  den = [1, 4.03e+07, 5.14e+14, 7.15e+18]
  system = signal.lti(num, den)

#  gradList = np.linspace(0.01, 0.16, num=16)
#  pcRadList = np.linspace(1.5, 2.9, num=15)
#  pcLenList = np.linspace(1.5, 2.9, num=15)

  detectorName = "P42574B"
  gradList =np.linspace(0.00, 0.001, 101)
  pcRad = 2.5
  pcLen = 1.6

#  detectorName = "P42662A"
#  gradList = np.linspace(0.025, 0.055, 101)
#  pcRad = 2
#  pcLen = 1.55

  filename = detectorName + "_fields_impgrad_%0.5f-%0.5f.npz" % (gradList[0], gradList[-1])

  wpArray  = None
  efld_rArray = None
  efld_zArray = None
  
  for (gradIdx,grad) in enumerate(gradList):
    print "on %d of %d" % (gradIdx, len(gradList))
    detName = "conf/%s_grad%0.5f_pcrad%0.2f_pclen%0.2f.conf" % (detectorName, grad,pcRad, pcLen)
    
    det =  Detector(detName, temperature=tempGuess, timeStep=1., numSteps=fitSamples*10, tfSystem=system)
    
    det.siggenInst.ReadElectricField()
    efld_r = np.array(det.siggenInst.GetElectricFieldR(), dtype=np.dtype('f4'), order='C')
    efld_z = np.array(det.siggenInst.GetElectricFieldZ(), dtype=np.dtype('f4'), order='C')
    #The "phi" component is always zero
#      efld_phi = np.array(det.siggenInst.GetElectricFieldPhi(), dtype=np.dtype('f4'), order='C')
    if efld_rArray is None:
      efld_rArray = np.zeros(  (efld_r.shape[0], efld_r.shape[1], len(gradList) ), dtype=np.dtype('f4'))
      efld_zArray = np.zeros(  (efld_z.shape[0], efld_z.shape[1], len(gradList) ), dtype=np.dtype('f4'))
      
    efld_rArray[:,:,gradIdx,] = efld_r
    efld_zArray[:,:,gradIdx,] = efld_z

  #WP doesn't change with impurity grad, so only need to loop thru pcRad
  wp = np.array(det.siggenInst.GetWeightingPotential(), dtype=np.dtype('f4'), order='C')
  if wpArray is None:
    wpArray = np.zeros((wp.shape[0], wp.shape[1], ), dtype=np.dtype('f4'))
  
  wpArray[:,:,] = wp

#  np.savez("P42574A_fields.npz", wpArray = wpArray, efld_rArray=efld_rArray, efld_zArray = efld_zArray, gradList = gradList, pcRadList = pcRadList  )

  r_space = np.arange(0, wpArray.shape[0]/10. , 0.1, dtype=np.dtype('f4'))
  z_space = np.arange(0, wpArray.shape[1]/10. , 0.1, dtype=np.dtype('f4'))

  efld_r_function = interpolate.RegularGridInterpolator((r_space, z_space, gradList, ), efld_rArray)
  efld_z_function = interpolate.RegularGridInterpolator((r_space, z_space, gradList, ), efld_zArray)

  np.savez(filename,  wpArray = wpArray, efld_rArray=efld_rArray, efld_zArray = efld_zArray, gradList = gradList,  pcRadList=None, pcLenList=None, wp_function = None, efld_r_function=efld_r_function, efld_z_function = efld_z_function, pcLen=pcLen, pcRad=pcRad  )


if __name__=="__main__":
    main(sys.argv[1:])


