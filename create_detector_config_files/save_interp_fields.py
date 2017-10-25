#!/usr/local/bin/python
import matplotlib
import sys, os

import numpy as np
from siggen_conf_generator.conf_file_interface import readConfigurationFile

import pandas as pd
from progressbar import ProgressBar, Percentage, Bar, ETA

def main(argv):

    # detectorName = "P42661A_bull"
    # pcRadiusRange = np.linspace(1.5,1.5,1)
    # pcLengthRange = np.linspace(1.7, 1.7,1)
    # gradientRange = np.linspace(-0.1,0.2,3)
    # impAvgRange = np.linspace(-0.65, -0.35, 3)

    detectorName = "P42574A_bull"
    pcRadiusRange = np.linspace(1.3,2.7,5)
    pcLengthRange = np.linspace(1.4,1.8,5)
    gradientRange = np.linspace(-0.05,0.1,3)
    impAvgRange = np.linspace(-1., -0.1, 3)

    # detectorName = "P42538A_bull"
    # pcRadiusRange = np.linspace(1.8,2.2,5)
    # pcLengthRange = np.linspace(1.4,1.8,5)
    # gradientRange = np.linspace(0,0.15,3)
    # impAvgRange = np.linspace(-0.6, -0.1, 3)
    #
    # detectorName = "P42661A_fine"
    # pcRadiusRange = np.linspace(1.5,1.6,2)
    # pcLengthRange = np.linspace(1.6,1.7,2)
    # gradientRange = np.linspace(0.0,0.05,2)
    # impAvgRange = np.linspace(-0.65, -0.35, 2)


    # detectorName = "P42661A_fine"
    # pcRadiusRange = [1.6]
    # pcLengthRange = [1.7]
    # gradientRange = [0.02]
    # impAvgRange = [-0.5]

    # print "grad step = %f" % (gradientRange[1] - gradientRange[0])
    # print "imp step = %f" % (impAvgRange[1] - impAvgRange[0])
    # # exit(0)

    shape_factor = 1
    filename = detectorName + "_fields.npz"

    wpArray  = None
    efld_rArray = None
    efld_zArray = None

    numfields = len(pcRadiusRange)*len(pcLengthRange)*len(gradientRange)*len(impAvgRange)
    bar = ProgressBar(widgets=[Percentage(), Bar(), ETA()], maxval=numfields).start()
    idx=0
    for (rad_idx, pcrad) in enumerate(pcRadiusRange):
      for (len_idx, pclen) in enumerate(pcLengthRange):
        for (gradIdx,grad) in enumerate(gradientRange):
          for (avgIdx, impAvg) in enumerate(impAvgRange):
            idx+=1
            bar.update(idx)
            detName = "conf/%s_grad%0.5f_avgimp%0.5f_pcrad%0.2f_pclen%0.2f.conf" % (detectorName, grad, impAvg, pcrad, pclen)

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
            nr = np.int(np.amax(r)*10/shape_factor+1)
            nz = np.int( np.amax(z)*10/shape_factor+1)
            efld_r = E_r.reshape(nr,nz)
            efld_z = E_z.reshape(nr,nz)

            data = pd.read_csv(wp_file, header=None, sep='\s+', skip_blank_lines=True, comment='#',
                    names=['r', 'z', 'wp'])
            r = data['r'].values
            z = data['z'].values
            wp = data['wp'].values
            wp = wp.reshape(nr,nz)

            if efld_rArray is None:
              efld_rArray = np.zeros(  (efld_r.shape[0], efld_r.shape[1],
                        len(gradientRange), len(impAvgRange), len(pcRadiusRange), len(pcLengthRange)
                        ), dtype=np.dtype('f4'), order="C" )
              efld_zArray = np.zeros(  (efld_z.shape[0], efld_z.shape[1],
                        len(gradientRange), len(impAvgRange), len(pcRadiusRange), len(pcLengthRange)
                        ), dtype=np.dtype('f4'), order="C" )
              wp_Array = np.zeros(  (efld_z.shape[0], efld_z.shape[1],
                        len(pcRadiusRange), len(pcLengthRange)
                        ), dtype=np.dtype('f4'), order="C" )

            # #turn them into c-contiguous style array of floats
            # efld_r = np.ascontiguousarray(efld_r, dtype=np.float32)
            # efld_z = np.ascontiguousarray(efld_z, dtype=np.float32)
            # wp = np.ascontiguousarray(wp, dtype=np.float32)

            efld_rArray[:,:,gradIdx,avgIdx, rad_idx, len_idx]=(efld_r)
            efld_zArray[:,:,gradIdx,avgIdx, rad_idx, len_idx]=(efld_z)
            wp_Array[:,:, rad_idx, len_idx]=(wp)

    bar.finish()
    np.savez(filename,  wpArray = np.ascontiguousarray(wp_Array), efld_rArray= np.ascontiguousarray(efld_rArray), efld_zArray =  np.ascontiguousarray(efld_zArray),
                        gradList = gradientRange, impAvgList =   impAvgRange,
                        pcLenList = pcLengthRange, pcRadList = pcRadiusRange
                  )

if __name__=="__main__":
    main(sys.argv[1:])
