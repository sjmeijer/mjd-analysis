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

import multiprocessing
from multiprocessing import Pool

ortec_spreadsheet = "ortec_ORTEC_Measurements.csv"


detectorName = "P42574A"
depletionVoltage = 1500.

def main():
  numThreads = multiprocessing.cpu_count()

  gradientRange = np.linspace(0.00, 0.001, 11)
  gradAvgRange = np.linspace(-0.565, -0.545, 11)

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
      for avg in gradAvgRange:
        startingFileName = detectorName + ".conf"
        newFileStr = copyConfFileWithNewImpurities(startingFileName, g, avg)
        print newFileStr

        runFieldgen(newFileStr)

        args.append( [ newFileStr] )

  pool = Pool(numThreads)
  pool.map(gen_conf_star, args)

def generateConfigFile(newFileStr):
  print newFileStr
  runFieldgen(newFileStr)

def gen_conf_star(a_b):
  return generateConfigFile(*a_b)

if __name__=="__main__":
    main()
