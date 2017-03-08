#!/usr/local/bin/python
import matplotlib
import sys, os

import numpy as np
from pysiggen import Detector

def main(argv):

  detectorName = "P42574A"
  gradList = np.linspace(0.00, 0.001, 21)
  impAvgList = np.linspace(-0.565, -0.545, 21)
  pcRad = 2.5
  pcLen = 1.6

  filename = detectorName + "_fields_impAndAvg_21by21.npz"

  wpArray  = None
  efld_rArray = None
  efld_zArray = None

  for (gradIdx,grad) in enumerate(gradList):
      for (avgIdx, impAvg) in enumerate(impAvgList):
            print "on %d of %d" % (gradIdx*len(impAvgList) + avgIdx, len(impAvgList)*len(impAvgList))


            detName = "conf/%s_grad%0.5f_avgimp%0.5f.conf" % (detectorName, grad, impAvg)

            det =  Detector(detName,  timeStep=1., numSteps=1000, )
            det.siggenInst.RunSiggenSetup()

            efld_r, efld_phi, efld_z, wp = det.siggenInst.ReadFields()
            # efld_r = np.array(det.siggenInst.GetElectricFieldR(), dtype=np.dtype('f4'), order='C')
            # efld_z = np.array(det.siggenInst.GetElectricFieldZ(), dtype=np.dtype('f4'), order='C')
            #The "phi" component is always zero
        #      efld_phi = np.array(det.siggenInst.GetElectricFieldPhi(), dtype=np.dtype('f4'), order='C')
            if efld_rArray is None:
              efld_rArray = np.zeros(  (efld_r.shape[0], efld_r.shape[1], len(gradList), len(impAvgList) ), dtype=np.dtype('f4'))
              efld_zArray = np.zeros(  (efld_z.shape[0], efld_z.shape[1], len(gradList), len(impAvgList) ), dtype=np.dtype('f4'))

            efld_rArray[:,:,gradIdx,avgIdx] = efld_r
            efld_zArray[:,:,gradIdx,avgIdx] = efld_z

            del det

  #WP doesn't change with impurity grad, so only need to loop thru pcRad
  # wp = np.array(det.siggenInst.GetWeightingPotential(), dtype=np.dtype('f4'), order='C')
  if wpArray is None:
    wpArray = np.zeros((wp.shape[0], wp.shape[1], ), dtype=np.dtype('f4'))

  wpArray[:,:,] = wp

  np.savez(filename,  wpArray = wpArray, efld_rArray=efld_rArray, efld_zArray = efld_zArray, gradList = gradList, impAvgList =   impAvgList,
                pcRadList=None, pcLenList=None, wp_function = None, efld_r_function=None, efld_z_function = None, pcLen=pcLen, pcRad=pcRad  )

if __name__=="__main__":
    main(sys.argv[1:])
