#!/usr/local/bin/python
import matplotlib
import sys, os

import numpy as np
from siggen_conf_generator.conf_file_interface import readConfigurationFile

import pandas as pd

def main(argv):

  detectorName = "P42574A"
  numGrads = 4
  numImps  = 4
  # gradList = np.linspace(0.00, 0.001, numGrads)
  # impAvgList = np.linspace(-0.565, -0.545, numImps)
  gradList = np.linspace(0, 0.02, numGrads)
  impAvgList = np.linspace(-0.55, -0.5, numImps)

  pcRad = 2.5
  pcLen = 1.6

  print "grad step = %f" % (gradList[1] - gradList[0])
  print "imp step = %f" % (impAvgList[1] - impAvgList[0])
  # exit(0)

  filename = detectorName + "_fields_impAndAvg_%dby%d.npz" % (numGrads, numImps)

  wpArray  = None
  efld_rArray = None
  efld_zArray = None
  # efld_rArray = np.empty((21,21),dtype=np.object)
  # efld_zArray = np.empty((21,21),dtype=np.object)

  for (gradIdx,grad) in enumerate(gradList):
      for (avgIdx, impAvg) in enumerate(impAvgList):
        print "on %d of %d" % (gradIdx*len(impAvgList) + avgIdx, len(impAvgList)*len(impAvgList))
        detName = "conf/%s_grad%0.5f_avgimp%0.5f.conf" % (detectorName, grad, impAvg)

        if not os.path.isfile(detName):
          print "Detector file %s not available" % detName
          exit(0)

        params = readConfigurationFile(detName)

        efld_file = params['field_name']
        wp_file = params['wp_name']

        if not os.path.isfile(efld_file):
          print "Field file %s not available" % efld_file
          exit(0)
        if not os.path.isfile(wp_file):
          print "WP file %s not available" % wp_file
          exit(0)

        data = pd.read_csv(efld_file, header=None, sep='\s+', skip_blank_lines=True, comment='#',
                names=['r', 'z', 'V', 'E', 'E_r', 'E_z'])
        r = data['r'].values
        z = data['z'].values
        E_r = data['E_r'].values
        E_z = data['E_z'].values
        nr = np.int(np.amax(r)*10+1)
        nz = np.int( np.amax(z)*10+1)
        efld_r = E_r.reshape(nr,nz)
        efld_z = E_z.reshape(nr,nz)

        data = pd.read_csv(wp_file, header=None, sep='\s+', skip_blank_lines=True, comment='#',
                names=['r', 'z', 'wp'])
        r = data['r'].values
        z = data['z'].values
        wp = data['wp'].values
        wp = wp.reshape(nr,nz)

        if efld_rArray is None:
          efld_rArray = np.zeros(  (efld_r.shape[0], efld_r.shape[1], len(gradList), len(impAvgList) ), dtype=np.dtype('f4'), order="C" )
          efld_zArray = np.zeros(  (efld_z.shape[0], efld_z.shape[1], len(gradList),len(impAvgList) ), dtype=np.dtype('f4'), order="C" )


        # #turn them into c-contiguous style array of floats
        # efld_r = np.ascontiguousarray(efld_r, dtype=np.float32)
        # efld_z = np.ascontiguousarray(efld_z, dtype=np.float32)
        wp = np.ascontiguousarray(wp, dtype=np.float32)

        efld_rArray[:,:,gradIdx,avgIdx]=(efld_r)
        efld_zArray[:,:,gradIdx,avgIdx]=(efld_z)

  np.savez(filename,  wpArray = wp, efld_rArray= np.ascontiguousarray(efld_rArray), efld_zArray =  np.ascontiguousarray(efld_zArray), gradList = gradList, impAvgList =   impAvgList,
                pcRadList=None, pcLenList=None, wp_function = None, efld_r_function=None, efld_z_function = None, pcLen=pcLen, pcRad=pcRad  )

if __name__=="__main__":
    main(sys.argv[1:])
