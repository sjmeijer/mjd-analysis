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

def main():
  numThreads = multiprocessing.cpu_count()

  gradientRange = np.linspace(0, 0.1, 31)
  gradAvgRange = np.linspace(-0.7, -0.5, 31)

  pcRadiusRange = [2.5]
  pcLengthRange = [1.7]

  startingFileName = "%s_bull.conf" % detectorName
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
        args.append( [ newFileStr] )

        # print "on %f,%f" % (g,avg)
        # runFieldgen(newFileStr)


  pool = Pool(numThreads)

  bar = ProgressBar(widgets=[Percentage(), Bar(), ETA()], maxval=len(args)).start()
  for i, _ in enumerate(pool.imap_unordered(gen_conf_star, args )):
      bar.update(i+1)
  bar.finish()
  pool.close()

def generateConfigFile(newFileStr):
  # print newFileStr
  runFieldgen(newFileStr)
  shutil.move(newFileStr, "conf/"+newFileStr)

def gen_conf_star(a_b):
  return generateConfigFile(*a_b)

if __name__=="__main__":
    main()