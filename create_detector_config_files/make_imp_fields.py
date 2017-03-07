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


  gradientRange = np.linspace(0.00, 0.02, 25)
  gradMultRange = np.linspace(1, 3, 25)
  gradPow = 3.
  
#  import matplotlib.pyplot as plt
#  plt.figure()
#  r0 = -10
#  for mult in np.linspace(1, 3, 11):
#    print mult
#    r = np.linspace(0, 33.7, 100)
#    z = np.empty_like(r)
#    z = r0 * (1 + (mult-1)*(r/33.7)**2)
#  
#    plt.plot(r,z)
#  
#  plt.show()
#  exit()

  print gradientRange
  
  pcRadiusRange = [2.5]#np.arange(2.5, 2.7, 0.05)
  pcLengthRange = [1.6]#np.arange(1.5, 1.75, 0.05)

  startingFileName = detectorName + ".conf"
  if not os.path.exists(startingFileName):
    print "The starting file %s does not exist." % startingFileName
    sys.exit()

  if not os.path.isdir("conf/fields"):
    print "Make a directory structure conf/fields for the field files to be dumped in..."
    sys.exit()

  args = []
  for g in gradientRange:
    startingFileName = detectorName + ".conf"
    newFileStart = detectorName + "_grad%0.4f" % g
    print newFileStart
    newFileStr = copyConfFileWithNewGradient(startingFileName, g)
    print newFileStr

    for mult in gradMultRange:
        newFileStr = copyConfFileWithNewRadialGradient(startingFileName, mult, gradPow, newFileStart)
        print newFileStr
        
        args.append( [ g,  newFileStr] )

  pool = Pool(numThreads)
  pool.map(gen_conf_star, args)

def generateConfigFile(g,  newFileStr):

  realImpurityZ0 = findImpurityZ0(newFileStr, depletionVoltage)
  writeFieldFiles(newFileStr, realImpurityZ0)

def gen_conf_star(a_b):
  return generateConfigFile(*a_b)

if __name__=="__main__":
    main()
