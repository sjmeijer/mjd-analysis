#!/usr/bin/python

'''
    Auto-generates siggen conf files
    Adjusts impurity info
    Generates field files

    B. Shanks, 5/6/15
    
'''

import sys, os

from siggen_conf_generator.parse_spreadsheets import *
from siggen_conf_generator.adjust_impurities import *

ortec_spreadsheet = "ortec_ORTEC_Measurements.csv"


def main():
    #parse in Alan's spreadsheet of ortec detectors
    detectorFile = open(ortec_spreadsheet, 'r')

    
    for line in detectorFile:
        crystalID = line.split(",")[0]
        
        #skip header lines (or anything that doesn't start with "P")
        if not crystalID.startswith("P"):
            continue
        
        print "\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
        print line
        print "Running with detector %s" % crystalID
        print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
        newImpurityZ0 = adjustDetectorByName(crystalID)
    
    detectorFile.close()


def adjustDetectorByName(crystalID):
    siggenInfo = read_alans_ortec_data(crystalID, "ortec_ORTEC_Measurements.csv", "ortec_starret_measurements.csv")
    confFileStr = siggenInfo.WriteToConfFile("ortec_default.config")
    
    if not os.path.exists(confFileStr):
        print "The file %s does not exist." % confFileStr
        sys.exit()

    desiredDepletion = siggenInfo.fDepletionVoltage

    realImpurityZ0 = findImpurityZ0(confFileStr, desiredDepletion)

    print "Determined the real impurity gradient to be %f" % realImpurityZ0

    writeFieldFiles(confFileStr, realImpurityZ0)

    return realImpurityZ0

if __name__=="__main__":
    main()


