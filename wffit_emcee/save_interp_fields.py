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

  gradList = np.linspace(0.01, 0.16, num=16)
  pcRadList = np.linspace(1.5, 2.9, num=15)
  pcLenList = np.linspace(1.5, 2.9, num=15)
  
  wpArray  = None
  efld_rArray = None
  efld_zArray = None
  
  for (radIdx, pcRad) in enumerate(pcRadList):
    for (lenIdx, pcLen) in enumerate(pcLenList):
      for (gradIdx,grad) in enumerate(gradList):
        detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (grad,pcRad, pcLen)
        
        det =  Detector(detName, temperature=tempGuess, timeStep=1., numSteps=fitSamples*10, tfSystem=system)
        
        det.siggenInst.ReadElectricField()
        efld_r = np.array(det.siggenInst.GetElectricFieldR(), dtype=np.dtype('f4'), order='C')
        efld_z = np.array(det.siggenInst.GetElectricFieldZ(), dtype=np.dtype('f4'), order='C')
        #The "phi" component is always zero
  #      efld_phi = np.array(det.siggenInst.GetElectricFieldPhi(), dtype=np.dtype('f4'), order='C')
        if efld_rArray is None:
          efld_rArray = np.zeros(  (efld_r.shape[0], efld_r.shape[1], len(gradList), len(pcRadList), len(pcLenList) ), dtype=np.dtype('f4'))
          efld_zArray = np.zeros(  (efld_z.shape[0], efld_z.shape[1], len(gradList), len(pcRadList), len(pcLenList)  ), dtype=np.dtype('f4'))
          
        efld_rArray[:,:,gradIdx, radIdx, lenIdx] = efld_r
        efld_zArray[:,:,gradIdx, radIdx, lenIdx] = efld_z

      #WP doesn't change with impurity grad, so only need to loop thru pcRad
      wp = np.array(det.siggenInst.GetWeightingPotential(), dtype=np.dtype('f4'), order='C')
      if wpArray is None:
        wpArray = np.zeros((wp.shape[0], wp.shape[1], len(pcRadList), len(pcLenList) ), dtype=np.dtype('f4'))
      
      wpArray[:,:,radIdx, lenIdx] = wp

#  np.savez("P42574A_fields.npz", wpArray = wpArray, efld_rArray=efld_rArray, efld_zArray = efld_zArray, gradList = gradList, pcRadList = pcRadList  )

  r_space = np.arange(0, wpArray.shape[0]/10. , 0.1, dtype=np.dtype('f4'))
  z_space = np.arange(0, wpArray.shape[1]/10. , 0.1, dtype=np.dtype('f4'))

  wp_function = interpolate.RegularGridInterpolator((r_space, z_space, pcRadList, pcLenList), wpArray)
  efld_r_function = interpolate.RegularGridInterpolator((r_space, z_space, gradList, pcRadList, pcLenList), efld_rArray)
  efld_z_function = interpolate.RegularGridInterpolator((r_space, z_space, gradList, pcRadList, pcLenList), efld_zArray)


  np.savez("P42574A_fields.npz",  wpArray = wpArray, efld_rArray=efld_rArray, efld_zArray = efld_zArray, gradList = gradList, pcRadList = pcRadList, pcLenList = pcLenList, wp_function = wp_function, efld_r_function=efld_r_function, efld_z_function = efld_z_function  )


if __name__=="__main__":
    main(sys.argv[1:])


