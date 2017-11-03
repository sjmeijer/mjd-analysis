#!/usr/local/bin/python
import matplotlib
import sys, os

import numpy as np
from siggen_conf_generator.conf_file_interface import readConfigurationFile

import pandas as pd

def main(argv):

  detectorName = "P42661A_bull_smallpc"

  gradientRange = np.linspace(0.08, 0.09, 3)
  impAvgRange = np.linspace(-0.52, -0.5, 3)

  numGrads = len(gradientRange)
  numImps  = len(impAvgRange)

  print "grad step = %f" % (gradientRange[1] - gradientRange[0])
  print "imp step = %f" % (impAvgRange[1] - impAvgRange[0])
  # exit(0)

  filename = detectorName + "_fields_%dby%d.npz" % (numGrads, numImps)

  wpArray  = None
  efld_rArray = None
  efld_zArray = None

  for (gradIdx,grad) in enumerate(gradientRange):
      for (avgIdx, impAvg) in enumerate(impAvgRange):
        print "on %d of %d" % (gradIdx*len(impAvgRange) + avgIdx, len(impAvgRange)*len(impAvgRange))
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
          efld_rArray = np.zeros(  (efld_r.shape[0], efld_r.shape[1], len(gradientRange), len(impAvgRange) ), dtype=np.dtype('f4'), order="C" )
          efld_zArray = np.zeros(  (efld_z.shape[0], efld_z.shape[1], len(gradientRange),len(impAvgRange) ), dtype=np.dtype('f4'), order="C" )


        # #turn them into c-contiguous style array of floats
        # efld_r = np.ascontiguousarray(efld_r, dtype=np.float32)
        # efld_z = np.ascontiguousarray(efld_z, dtype=np.float32)
        wp = np.ascontiguousarray(wp, dtype=np.float32)

        efld_rArray[:,:,gradIdx,avgIdx]=(efld_r)
        efld_zArray[:,:,gradIdx,avgIdx]=(efld_z)

  np.savez(filename,  wpArray = wp, efld_rArray= np.ascontiguousarray(efld_rArray), efld_zArray =  np.ascontiguousarray(efld_zArray),
                        gradList = gradientRange, impAvgList =   impAvgRange,
                  )

if __name__=="__main__":
    main(sys.argv[1:])
