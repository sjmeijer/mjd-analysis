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


def main():

#  detectorName = "P42664A"
#  depletionVoltage = 600.
  detectorName = "P42574A"
  depletionVoltage = 1500.
  
  gradientRange = np.arange(0.05, 0.07, 0.0025)
  pcRadiusRange = np.arange(2.35, 2.5
  5, 0.025)
  
  startingFileName = detectorName + ".conf"


  siggenParams = readConfigurationFile(startingFileName)

  for grad in gradientRange:
    confFileName = "P42574A_grad%0.3f_final.conf" % grad
    if not os.path.exists(confFileName):
        print "The starting file %s does not exist." % confFileName
        continue
  
    newFileStart = detectorName + "_grad%0.3f" % grad
    for rad in pcRadiusRange:
    

  
      newFileStr = copyConfFileWithNewPcRadius(confFileName, rad, newFileStart)
    
      runFieldgen(newFileStr)


if __name__=="__main__":
    main()


