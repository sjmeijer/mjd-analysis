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
#
# detectorName = "P42538A_bull"
# pcRadiusRange = np.linspace(1.8,2.2,5)
# pcLengthRange = np.linspace(1.4,1.8,5)
# gradientRange = np.linspace(0,0.15,3)
# impAvgRange = np.linspace(-0.6, -0.1, 3)

# detectorName = "P42574A_bull"
# pcRadiusRange = np.linspace(1.3,2.7,5)
# pcLengthRange = np.linspace(1.4,1.8,5)
# gradientRange = np.linspace(-0.05,0.1,3)
# impAvgRange = np.linspace(-1., -0.1, 3)

# detectorName = "P42661A_bull"
# pcRadiusRange = np.linspace(1.53778808688,1.53778808688,1)
# pcLengthRange = np.linspace(1.21040959774, 1.21040959774,1)
# gradientRange = np.linspace(0.0142337933795,0.0142337933795,1)
# impAvgRange = np.linspace(-0.545287876729, -0.545287876729, 1)

detectorName = "P42661A_long"
pcRadiusRange = np.linspace(1.5,1.6,2)
pcLengthRange = np.linspace(1.6,1.7,2)
gradientRange = np.linspace(0.0,0.05,2)
impAvgRange = np.linspace(-0.65, -0.35, 2)

# detectorName = "P42661A_bull"
# pcRadiusRange = [1.6]
# pcLengthRange = [1.7]
# gradientRange = [0.02]
# impAvgRange = [-0.5]

def main():
  numThreads = multiprocessing.cpu_count()

  startingFileName = detectorName + ".conf"
  if not os.path.exists(startingFileName):
    print "The starting file %s does not exist." % startingFileName
    sys.exit()

  if not os.path.isdir("conf/fields"):
    print "Make a directory called fields for the field files to be dumped in..."
    sys.exit()

  args = []
  for g in gradientRange:
    for avg in impAvgRange:
      for r in pcRadiusRange:
        for l in pcLengthRange:
            wp_name = "conf/fields/" + detectorName +"_grad%0.5f_avgimp%0.5f_pcrad%0.2f_pclen%0.2f_wp.dat" % (g, avg, r, l)
            if os.path.exists(wp_name):
                print "skipping: ", g, avg, r, l
                continue

            newFileStr = copyConfFileWithNewEverything(startingFileName, g, avg, r, l)
            args.append( [ newFileStr] )

            # generateConfigFile(newFileStr)

#        field_name = "fields/" + detectorName +"_grad%0.2f_pcrad%0.2f_pclen%0.2f_ev.dat" % (g, r, l)
#        if not os.path.exists(field_name):

  if len(args) >= 1:
      pool = Pool(numThreads)

      bar = ProgressBar(widgets=[Percentage(), Bar(), ETA()], maxval=len(args)).start()
      for i, _ in enumerate(pool.imap_unordered(gen_conf_star, args )):
          bar.update(i+1)
      bar.finish()
      pool.close()

  pool = Pool(numThreads)
  pool.map(gen_conf_star, args)

def generateConfigFile( newFileStr):
  runFieldgen(newFileStr)
  shutil.move(newFileStr, "conf/"+newFileStr)
  # realImpurityZ0 = findImpurityZ0(newFileStr, depletionVoltage)
  # writeFieldFiles(newFileStr, realImpurityZ0)

def gen_conf_star(a_b):
  return generateConfigFile(*a_b)

if __name__=="__main__":
    main()
