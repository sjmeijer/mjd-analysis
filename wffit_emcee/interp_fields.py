#!/usr/local/bin/python
from ROOT import *

import matplotlib
#matplotlib.use('CocoaAgg')
import sys, os
import matplotlib.pyplot as plt
import scipy.optimize as op


import numpy as np

from scipy import signal, interpolate

from detector_model import *

from plot_wp_and_efld import plotWP, plotEF

def main(argv):

  plt.ion()
  
  fitSamples = 250 #has to be longer than the longest wf you're gonna fit
  tempGuess = 81.5
  
  #Set up detectors
  num = [3.64e+09, 1.88e+17, 6.05e+15]
  den = [1, 4.03e+07, 5.14e+14, 7.15e+18]
  system = signal.lti(num, den)

  gradList = [0.04, 0.08]
  pcRadList = [2.25, 2.75]#this is the actual starret value ish
  detArray  = np.empty( (len(gradList),len(pcRadList)), dtype=object)
  
  wpArray  = None
  efld_rArray = None
  efld_zArray = None
  
  for (radIdx, pcRad) in enumerate(pcRadList):
    for (gradIdx,grad) in enumerate(gradList):
      detName = "conf/P42574A_grad%0.2f_pcrad%0.2f.conf" % (grad,pcRad)
      det =  Detector(detName, temperature=tempGuess, timeStep=1., numSteps=fitSamples*10, tfSystem=system)
      detArray[gradIdx, radIdx] = det

      det.siggenInst.ReadElectricField()
      efld_r = np.array(det.siggenInst.GetElectricFieldR(), dtype=np.dtype('f4'), order='C')
      efld_z = np.array(det.siggenInst.GetElectricFieldZ(), dtype=np.dtype('f4'), order='C')
      #The "phi" component is always zero
#      efld_phi = np.array(det.siggenInst.GetElectricFieldPhi(), dtype=np.dtype('f4'), order='C')
      if efld_rArray is None:
        efld_rArray = np.zeros(  (efld_r.shape[0], efld_r.shape[1], len(gradList), len(pcRadList) ), dtype=np.dtype('f4'))
        efld_zArray = np.zeros(  (efld_z.shape[0], efld_z.shape[1], len(gradList), len(pcRadList) ), dtype=np.dtype('f4'))
        
      efld_rArray[:,:,gradIdx, radIdx] = efld_r
      efld_zArray[:,:,gradIdx, radIdx] = efld_z

    #WP doesn't change with impurity grad, so only need to loop thru pcRad
    wp = np.array(det.siggenInst.GetWeightingPotential(), dtype=np.dtype('f4'), order='C')
    if wpArray is None:
      wpArray = np.zeros((wp.shape[0], wp.shape[1], len(pcRadList)), dtype=np.dtype('f4'))
    
    wpArray[:,:,radIdx] = wp


  print wpArray.shape
  print efld_rArray.shape
  print efld_zArray.shape

  r_space = np.arange(0, wpArray.shape[0]/10. , 0.1, dtype=np.dtype('f4'))
  z_space = np.arange(0, wpArray.shape[1]/10. , 0.1, dtype=np.dtype('f4'))

  wp_function = interpolate.RegularGridInterpolator((r_space, z_space, pcRadList), wpArray)
  efld_r_function = interpolate.RegularGridInterpolator((r_space, z_space, gradList, pcRadList), efld_rArray)
  efld_z_function = interpolate.RegularGridInterpolator((r_space, z_space, gradList, pcRadList), efld_zArray)

  (rr, zz) = np.meshgrid(r_space, z_space)
  pcpc = np.ones_like(rr) * 2.5
  gradgrad = np.ones_like(rr) * 0.06

  points_wp =  np.array([rr.flatten() , zz.flatten(), pcpc.flatten()], dtype=np.dtype('f4') ).T
  points_ef =  np.array([rr.flatten() , zz.flatten(), gradgrad.flatten(), pcpc.flatten()], dtype=np.dtype('f4') ).T
  
  new_wp = np.array(wp_function( points_wp ).reshape(rr.shape).T, dtype=np.dtype('f4'), order="C")
  new_ef_r = np.array(efld_r_function( points_ef ).reshape(rr.shape).T, dtype=np.dtype('f4'), order="C")
  new_ef_z = np.array(efld_z_function( points_ef ).reshape(rr.shape).T, dtype=np.dtype('f4'), order="C")
#
  print new_wp.shape
  print new_ef_r.shape
  print new_ef_z.shape
  
  plotWPNoDet(new_wp)
  plotEFNoDet(new_ef_r, new_ef_z)

  wp_pp = getPointer(new_wp)
  efr_pp = getPointer(new_ef_r)
  efz_pp = getPointer(new_ef_z)

  det.siggenInst.SetWeightingPotential( wp_pp )
  det.siggenInst.SetElectricField( efr_pp, efz_pp )
  
  f1 = plt.figure(3)
  f2 = plt.figure(4)
  
  plotWP(det, f1)
  plotEF(det, f2)


  value = raw_input('  --> Press q to quit, any other key to continue\n')


def plotWPNoDet(wp):

  plt.figure()

  plt.imshow(wp.T, origin='lower',  interpolation='nearest', cmap=plt.cm.RdYlBu_r)
  
  plt.title("WP from siggen memory")
  plt.xlabel("radial (mm)")
  plt.ylabel("axial (mm)")
  
#  plt.xlim(0,5)
#  plt.ylim(0,5)

def plotEFNoDet(efld_r, efld_z):

  plt.figure()
  mag = np.sqrt( np.add(np.square(efld_r), np.square(efld_z)) )

#  mag[np.where(mag==0)] = np.nan

  plt.imshow(mag.T, origin='lower', interpolation='nearest', cmap=plt.cm.RdYlBu_r)
  
  plt.title("E field from siggen memory")
  plt.xlabel("radial (mm)")
  plt.ylabel("axial (mm)")

def getPointer(floatfloat):
  return (floatfloat.__array_interface__['data'][0] + np.arange(floatfloat.shape[0])*floatfloat.strides[0]).astype(np.intp)



if __name__=="__main__":
    main(sys.argv[1:])


