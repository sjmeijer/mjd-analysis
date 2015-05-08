#!/usr/bin/python

import sys
import array
import cmath
from ctypes import c_ulonglong
import os
import subprocess
import math
import re
import time

from conf_file_interface import SiggenCrystalInfo

'''
Reads in various detector info spreadsheets, spits out siggen config files

VERY specifically engineered for the task at hand

Requires some default config file template.

B. Shanks, 5/5/15

'''


impurity_grad_min = 0.04


#reads in alan's ortec data from .csv files generated from the google spreadsheet
# https://docs.google.com/spreadsheets/d/1ONxi-gF6ccdkqF6d2W0zujPkNA0UbruEej-ESQlxIg4/edit#gid=0
# spits back a dict with things one would expect to want to put into a siggen conf file
def read_alans_ortec_data(crystalID, ortecMeasurementFile, starrettMeasurementFile):

    # read in the ortec measurements
    ortecFile = open(ortecMeasurementFile, 'r')
    for line in ortecFile:
        if line.find(crystalID) >=0:
            ortecDict = parse_alan_ortec_measurement_line(line)
            break
    else:
        print "Crystal ID %s not found in %s" % (crystalID, ortecMeasurementFile)
        sys.exit()
    ortecFile.close()

    # read in the ortec measurements
    starrettFile = open(starrettMeasurementFile, 'r')
    for line in starrettFile:
        if line.find(crystalID) >=0:
            starDict = parse_alan_starrett_measurement_line(line)
            break
    else:
        print "Crystal ID %s not found in %s" % (crystalID, starrettMeasurementFile)
        sys.exit()
    starrettFile.close()

    siggenInfo = SiggenCrystalInfo()

    siggenInfo.SetCrystalType( 'ortec' )
    siggenInfo.fCrystalID = ortecDict['xtal_id']

    #use starret measurements
    try:
        siggenInfo.siggen_xtal_radius = float( starDict['xtal_diameter'] ) / 2
        siggenInfo.siggen_xtal_length = float( starDict['xtal_length'] )
    except ValueError:
        print "Detector must have all starret measurements.  Diameter or length is unreadable."
        sys.exit()
    #pull in the mandatory ortec measurements
    try:
        siggenInfo.siggen_pc_radius = float( ortecDict['pc_diameter'] ) / 2
        siggenInfo.siggen_pc_length = float( ortecDict['pc_depth'] )
    except ValueError:
        print "Detector must have point contact measurements. PC Diameter or depth is ureadable"
        sys.exit()
    try:
        siggenInfo.fDepletionVoltage = float( ortecDict['depletion_v'] )
        siggenInfo.siggen_xtal_HV = float( ortecDict['operating_v'] )
    except ValueError:
        print "Detector must have bias and operating voltages.  No info found."
        sys.exit()

    #pull in optional ortec measurements
    try:
        siggenInfo.siggen_Li_thickness = float( ortecDict['dl_depth'] ) / 1000
    except ValueError:
        print "No dead layer thickness found.  Using default value..."

    try:
        impurity_tail = float( ortecDict['impurity_tail'] )
        impurity_seed = float( ortecDict['impurity_seed'] )
    except ValueError:
        print "No impurity information found.  Using default values..."

    #convert impurity information to siggen type

    #choose highest impurity side as z0 impurity.
    #convert from 10^9 to 10^10 e/cm^3, make negative for siggen p type flag
    impurity_z0 = -1 * max(impurity_tail, impurity_seed) / 10

    impurity_delta = abs(impurity_tail - impurity_seed) / 10

    #in 10^10 e/cm^4
    impurity_grad = impurity_delta / (siggenInfo.siggen_xtal_length/10) #convert length to cm

    #check to make sure the gradient is something reasonable

    if impurity_grad < impurity_grad_min:
        print "   Adjusting impurity gradient upwards from %f to %f\n" % (impurity_grad, impurity_grad_min)
        impurity_grad = impurity_grad_min

    siggenInfo.siggen_impurity_z0 = impurity_z0
    siggenInfo.siggen_impurity_gradient = impurity_grad

    return siggenInfo

def parse_alan_ortec_measurement_line(line):
    #which column is what?
    ORTEC_CRYSTAL_ID_COL    = 0
    ORTEC_DIAMETER_COL      = 1  #in mm
    ORTEC_LENGTH_COL        = 2  #in mm
    ORTEC_PC_DIAMETER_COL   = 8  #in mm
    ORTEC_PC_DEPTH_COL      = 9  #in mm
    ORTEC_DL_DEPTH_COL      = 10 #in um
    ORTEC_IMPURITY_TAIL_COL = 11 #in e9
    ORTEC_IMPURITY_SEED_COL = 12 #in e9
    ORTEC_DEPLETION_V_COL   = 13 #in e9
    ORTEC_OPERATING_V_COL   = 14 #in e9
    
    #assume its a csv
    lineCol = line.split(",")
    #read in value by value
    #unprotected casts to float.  will crash if value isn't a number (except for the ID)
    valueDict = {}
    valueDict['xtal_id']         = lineCol[ORTEC_CRYSTAL_ID_COL]
    valueDict['xtal_diameter']   = float( lineCol[ORTEC_DIAMETER_COL]       )
    valueDict['xtal_length']     = float( lineCol[ORTEC_LENGTH_COL]         )
    valueDict['pc_diameter']     = float( lineCol[ORTEC_PC_DIAMETER_COL]    )
    valueDict['pc_depth']        = float( lineCol[ORTEC_PC_DEPTH_COL]       )
    valueDict['dl_depth']        = float( lineCol[ORTEC_DL_DEPTH_COL]       )
    valueDict['impurity_tail']   = float( lineCol[ORTEC_IMPURITY_TAIL_COL]  )
    valueDict['impurity_seed']   = float( lineCol[ORTEC_IMPURITY_SEED_COL]  )
    valueDict['depletion_v']     = float( lineCol[ORTEC_DEPLETION_V_COL]    )
    valueDict['operating_v']     = float( lineCol[ORTEC_OPERATING_V_COL]    )

    return valueDict

def parse_alan_starrett_measurement_line(line):
    #which column is what?
    STARRETT_CRYSTAL_ID_COL  = 0
    STARRETT_DIAMETER_COL    = 6  #in mm
    STARRETT_LENGTH_COL      = 7  #in mm

    #assume its a csv
    lineCol = line.split(",")
    #read in value by value
    #unprotected casts to float.  will crash if value isn't a number (except for the ID)
    valueDict = {}
    valueDict['xtal_id']         = lineCol[STARRETT_CRYSTAL_ID_COL]
    valueDict['xtal_diameter']   = float( lineCol[STARRETT_DIAMETER_COL]       )
    valueDict['xtal_length']     = float( lineCol[STARRETT_LENGTH_COL]         )

    return valueDict

