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
from progressbar import ProgressBar, Percentage, Bar, ETA

ortec_spreadsheet = "ortec_ORTEC_Measurements.csv"


detectorName = "P42574A"
depletionVoltage = 1500.

def main():
  numThreads = multiprocessing.cpu_count()

  gradientRange = np.linspace(0, 0.02, 4)
  gradAvgRange = np.linspace(-0.55, -0.5, 4)

  pcRadiusRange = [2.5]
  pcLengthRange = [1.7]

  startingFileName = detectorName +  ".conf"
  if not os.path.exists(startingFileName):
    print "The starting file %s does not exist." % startingFileName
    sys.exit()
  if not os.path.isdir("conf/fields"):
    print "Make a directory structure conf/fields for the field files to be dumped in..."
    sys.exit()

  args = []
  for g in gradientRange:
      for avg in gradAvgRange:
        newFileStr = copyConfFileWithNewImpurities(startingFileName, g, avg)

        # runFieldgen(newFileStr)

        args.append( [ newFileStr] )

  pool = Pool(numThreads)

  bar = ProgressBar(widgets=[Percentage(), Bar(), ETA()], maxval=len(args)).start()
  for i, _ in enumerate(pool.imap_unordered(gen_conf_star, args )):
      bar.update(i+1)
  bar.finish()
  pool.close()

def generateConfigFile(newFileStr):
  # print newFileStr
  runFieldgen(newFileStr)

def gen_conf_star(a_b):
  return generateConfigFile(*a_b)

if __name__=="__main__":
    main()
