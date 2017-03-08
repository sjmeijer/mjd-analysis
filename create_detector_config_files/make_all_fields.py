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


#detectorName = "P42574A"
#depletionVoltage = 1500.
#gradientRange = np.linspace(0.00, 0.001, 101)
#pcRadiusRange = [2.5]#np.arange(2.5, 2.7, 0.05)
#pcLengthRange = [1.6]#np.arange(1.5, 1.75, 0.05)


detectorName = "P42662A"
depletionVoltage = 2000
pcRadiusRange = [2.]#np.arange(2.5, 2.7, 0.05)
pcLengthRange = [1.55]#np.arange(1.5, 1.75, 0.05)
gradientRange = np.linspace(0.025, 0.055, 101)

def main():
  numThreads = 4


  startingFileName = detectorName + ".conf"
  if not os.path.exists(startingFileName):
    print "The starting file %s does not exist." % startingFileName
    sys.exit()

  if not os.path.isdir("conf/fields"):
    print "Make a directory called fields for the field files to be dumped in..."
    sys.exit()

  args = []
  for g in gradientRange:
    startingFileName = detectorName + ".conf"
    newFileStart = detectorName + "_grad%0.5f" % g
    print newFileStart
    newFileStr = copyConfFileWithNewGradient(startingFileName, g)
    print newFileStr

    for r in pcRadiusRange:
      for l in pcLengthRange:
        newFileStr = copyConfFileWithNewPcRadius(newFileStr, r, l, newFileStart)
        print newFileStr

#        generateConfigFile(g, r, l, newFileStr)

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
