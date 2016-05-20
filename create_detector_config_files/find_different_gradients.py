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

ortec_spreadsheet = "ortec_ORTEC_Measurements.csv"


def main():

#  detectorName = "P42664A"
#  depletionVoltage = 600.
  detectorName = "P42574A"
  depletionVoltage = 1500.
  
#  gradientRange = np.arange(0.01, 0.2, 0.01)
#  pcRadiusRange = np.arange(1.65, 2.75, 0.1)
  gradientRange = np.arange(0.05, 0.07, 0.0025)

  startingFileName = detectorName + ".conf"
  if not os.path.exists(startingFileName):
    print "The starting file %s does not exist." % startingFileName
    sys.exit()

  siggenParams = readConfigurationFile(startingFileName)

  for grad in gradientRange:
    newFileStr = copyConfFileWithNewGradient(startingFileName, grad)
    
#    print ">>>new file string is %s" % newFileStr
#
#
#    configFileSplit = newFileStr.split(".")
#    finalConfigFileStr = '.'.join(configFileSplit[:-1]) + "_final." +  configFileSplit[-1]
#    print ">>>final string will be %s" % finalConfigFileStr

    realImpurityZ0 = findImpurityZ0(newFileStr, depletionVoltage)
    
    
    print "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&"
    print "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&"
    print ">>for %s with gradient %0.2f the final impurity z0 is %0.3f" % (detectorName, grad, realImpurityZ0)
    print "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&"
    print "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&"
    
    writeFieldFiles(newFileStr, realImpurityZ0)


if __name__=="__main__":
    main()


