#!/usr/bin/python

'''
    Auto-generates siggen conf files
    Adjusts impurity info
    Generates field files

    B. Shanks, 5/6/15
    
'''

import sys, os
import numpy as np

from siggen_conf_generator.parse_spreadsheets import *
from siggen_conf_generator.adjust_impurities import *

from multiprocessing import Pool

ortec_spreadsheet = "ortec_ORTEC_Measurements.csv"


detectorName = "P42574A"
depletionVoltage = 1500.

def main():
  numThreads = 4

#  detectorName = "P42664A"
#  depletionVoltage = 600.

  
  gradientRange = np.arange(0.04, 0.09, 0.01)
  pcRadiusRange = np.arange(2.4, 2.7, 0.05)
  pcLengthRange = np.arange(1.5, 1.75, 0.05)

  startingFileName = detectorName + ".conf"
  if not os.path.exists(startingFileName):
    print "The starting file %s does not exist." % startingFileName
    sys.exit()

  args = []
  for g in gradientRange:
    startingFileName = detectorName + ".conf"
    newFileStart = detectorName + "_grad%0.2f" % g
    print newFileStart
    newFileStr = copyConfFileWithNewGradient(startingFileName, g)
    print newFileStr
  
    for r in pcRadiusRange:
      for l in pcLengthRange:
        newFileStr = copyConfFileWithNewPcRadius(newFileStr, r, l, newFileStart)
        print newFileStr

        #generateConfigFile(g, r, l)

#        field_name = "fields/" + detectorName +"_grad%0.2f_pcrad%0.2f_pclen%0.2f_ev.dat" % (g, r, l)
#        if not os.path.exists(field_name):
        args.append( [ g, r, l, newFileStr] )

  pool = Pool(numThreads)
  pool.map(gen_conf_star, args)

def generateConfigFile( grad, pcRad, pcLen, newFileStr):
  
  realImpurityZ0 = findImpurityZ0(newFileStr, depletionVoltage)
  writeFieldFiles(newFileStr, realImpurityZ0)

def gen_conf_star(a_b):
  return generateConfigFile(*a_b)

if __name__=="__main__":
    main()


