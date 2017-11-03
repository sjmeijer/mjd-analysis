#!/usr/bin/python

'''
    Python translation of bash script by J. Nance, Spring 2015.

    Changed search routines to golden section method.

    B. Shanks, 5/4/15

'''

import sys
import array
import cmath
from ctypes import c_ulonglong
import os, shutil
import subprocess
import math
import re
import time

from conf_file_interface import *

verbose = 0



'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

'''Setup options'''

mjd_siggen_dir = "~/Dev/mjd_siggen/"
#mjd_siggen_dir = "~/Dev/siggen/mjd_siggen/"
grid_size = 0.5 #fielgen grid size.  usually 0.1 or 0.5.  0.5 is WAY faster
depletion_tolerance = 10 #guaranteed to be within this tolerance


#define the golden ratio so I don't require scipy to do golden section search
golden = (1 + 5 ** 0.5) / 2
'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''


def main(argv):

    #read in first argument as the file to find z0 on
    configFileStr = argv[0]
    configFileStr = os.path.expanduser(configFileStr)
    if not os.path.exists(configFileStr):
        print "The file %s does not exist." % configFileStr
        sys.exit()

    #read in the second argument as the desired depletion voltage
    desiredDepletion = float(argv[1])

    realImpurityZ0 = findImpurityZ0(configFileStr, desiredDepletion)

    print "Determined the real impurity z0 to be %f" % realImpurityZ0

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

def copyConfFileWithNewGradient(fileName, newGradient):
    configFileSplit = fileName.split(".")
    appendString = "_grad%0.5f." % newGradient
    newConfigFileStr = configFileSplit[0] + appendString +  configFileSplit[1]
    print "cp %s %s" % (fileName, newConfigFileStr)
    #os.rename(fileName, newConfigFileStr )
    shutil.copy(fileName, newConfigFileStr)
    replaceConfFileValue(newConfigFileStr, 'impurity_gradient', newGradient)

    field_name= "conf/fields/" + configFileSplit[0] + ("_grad%0.5f" %newGradient) + "_"

    replaceConfFileValue(newConfigFileStr, 'field_name', field_name + "ev.dat")
    replaceConfFileValue(newConfigFileStr, 'wp_name', field_name + "wp.dat")

    return newConfigFileStr

def copyConfFileWithNewGradAndZ0(fileName, newGradient, new_z0):


    # print "avg imp: %f, imp z0: %f" % (newAvgImpurity, new_z0)

    configFileSplit = fileName.split(".")
    appendString = "_grad%0.5f_z0%0.5f" % (newGradient, new_z0)
    newConfigFileStr = configFileSplit[0] + appendString +  "." + configFileSplit[1]

    if verbose:
        print "cp %s %s" % (fileName, newConfigFileStr)

    #os.rename(fileName, newConfigFileStr )
    shutil.copy(fileName, newConfigFileStr)
    replaceConfFileValue(newConfigFileStr, 'impurity_gradient', newGradient)
    replaceConfFileValue(newConfigFileStr, 'impurity_z0', new_z0)

    field_name= "conf/fields/" + configFileSplit[0] + appendString + "_"
    replaceConfFileValue(newConfigFileStr, 'field_name', field_name + "ev.dat")
    replaceConfFileValue(newConfigFileStr, 'wp_name', field_name + "wp.dat")

    return newConfigFileStr

def copyConfFileWithNewImpurities(fileName, newGradient, newAvgImpurity):

    oldParams = readConfigurationFile(fileName)
    det_z = oldParams['xtal_length']/10. # in cm
    if det_z == 0:
        print "zero det_Z: some bug is here"
        exit(0)

    new_z0 = newAvgImpurity - newGradient*det_z/2.
    configFileSplit = fileName.split(".")
    appendString = "_grad%0.5f_avgimp%0.5f" % (newGradient, newAvgImpurity)
    newConfigFileStr = configFileSplit[0] + appendString +  "." + configFileSplit[1]

    if verbose:
        print "cp %s %s" % (fileName, newConfigFileStr)

    #os.rename(fileName, newConfigFileStr )
    shutil.copy(fileName, newConfigFileStr)
    replaceConfFileValue(newConfigFileStr, 'impurity_gradient', newGradient)
    replaceConfFileValue(newConfigFileStr, 'impurity_z0', new_z0)

    field_name= "conf/fields/" + configFileSplit[0] + appendString + "_"
    replaceConfFileValue(newConfigFileStr, 'field_name', field_name + "ev.dat")
    replaceConfFileValue(newConfigFileStr, 'wp_name', field_name + "wp.dat")
    return newConfigFileStr
    '''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''
def copyConfFileWithNewEverything(fileName, newGradient, newAvgImpurity, newpcRadius, newpcLength):

    oldParams = readConfigurationFile(fileName)
    det_z = oldParams['xtal_length']/10. # in cm
    if det_z == 0:
        print "zero det_Z: some bug is here"
        exit(0)

    new_z0 = newAvgImpurity - newGradient*det_z/2.
    configFileSplit = fileName.split(".")
    appendString = "_grad%0.5f_avgimp%0.5f_pcrad%0.2f_pclen%0.2f" % (newGradient, newAvgImpurity,newpcRadius, newpcLength)

    newConfigFileStr = configFileSplit[0] + appendString +  "." + configFileSplit[1]

    if verbose:
        print "cp %s %s" % (fileName, newConfigFileStr)

    #os.rename(fileName, newConfigFileStr )
    shutil.copy(fileName, newConfigFileStr)
    replaceConfFileValue(newConfigFileStr, 'impurity_gradient', newGradient)
    replaceConfFileValue(newConfigFileStr, 'impurity_z0', new_z0)
    replaceConfFileValue(newConfigFileStr, 'pc_length', newpcLength)
    replaceConfFileValue(newConfigFileStr, 'pc_radius', newpcRadius)

    field_name= "conf/fields/" + configFileSplit[0] + appendString + "_"
    replaceConfFileValue(newConfigFileStr, 'field_name', field_name + "ev.dat")
    replaceConfFileValue(newConfigFileStr, 'wp_name', field_name + "wp.dat")
    return newConfigFileStr
    '''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

def copyConfFileWithNewRadialGradient(fileName, gradMult, gradPow, newFileStart=None):
    configFileSplit = fileName.split(".")
    appendString = "_radmult%0.1f_pow%d" % (gradMult, gradPow)

    if newFileStart is None:
      newFileStart = configFileSplit[:-1]

    newConfigFileStr = newFileStart + appendString + "." + configFileSplit[-1]

    print "cp %s %s" % (fileName, newConfigFileStr)
    #os.rename(fileName, newConfigFileStr )
    shutil.copy(fileName, newConfigFileStr)

    replaceConfFileValue(newConfigFileStr, 'impurity_radial_mult', gradMult)
    replaceConfFileValue(newConfigFileStr, 'impurity_rpower', gradPow)

    field_name= "conf/fields/" + newFileStart + appendString + "_"

    replaceConfFileValue(newConfigFileStr, 'field_name', field_name + "ev.dat")
    replaceConfFileValue(newConfigFileStr, 'wp_name', field_name + "wp.dat")

    return newConfigFileStr
'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

def copyConfFileWithNewPcRadius(fileName, newRadius, newLength, newFileStart=None):
    configFileSplit = fileName.split(".")

    appendString = "_pcrad%0.2f_pclen%0.2f" % (newRadius, newLength)
    if newFileStart is None:
      newFileStart = configFileSplit[:-1]

    newConfigFileStr = newFileStart + appendString + "." + configFileSplit[-1]

    print "cp %s %s" % (fileName, newConfigFileStr)
    #os.rename(fileName, newConfigFileStr )
    shutil.copy(fileName, newConfigFileStr)

    replaceConfFileValue(newConfigFileStr, 'pc_length', newLength)
    replaceConfFileValue(newConfigFileStr, 'pc_radius', newRadius)

    field_name= "conf/fields/" + newFileStart + appendString + "_"

    replaceConfFileValue(newConfigFileStr, 'field_name', field_name + "ev.dat")
    replaceConfFileValue(newConfigFileStr, 'wp_name', field_name + "wp.dat")

    return newConfigFileStr
'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

def writeFieldFiles(fileName, impurityZ0):
#    #create a test file for changing the impurity gradient
#    configFileSplit = fileName.split(".")
##    appendString = "_grad%0.2f." % impurityZ0
#
#    finalConfigFileStr = '.'.join(configFileSplit[:-1]) + "_final." +  configFileSplit[-1]
#    #finalConfigFileStr = configFileSplit[:-2] + "_final." +  configFileSplit[-1]
#    os.system(  "cp %s %s" % (fileName, finalConfigFileStr) )
#
    #change the impurity grad in the copied config file
    replaceConfFileValue(fileName, 'impurity_z0', impurityZ0)

    runFieldgen(fileName)
'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''


def runFieldgen(fileName):
      #run fieldgen, save field files
    fieldGenDir = os.path.expanduser(mjd_siggen_dir)
    args = [fieldGenDir + "mjd_fieldgen", '-c', fileName, "-w", "1", "-p", "1"]
    output = subprocess.check_output(args)

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

def findImpurityZ0(fileName, desiredDepletion):
    #read in original parameters
    #originalParams = paramStructure#readConfigurationFile(fileName)
    originalParams = readConfigurationFile(fileName)
    original_impurity_z0 = originalParams['impurity_z0']
    original_bias = originalParams['bias']
    crystal_length = originalParams['xtal_length']
    impurity_grad = originalParams['impurity_grad']

    print ""
    print "Original params: bias=%d Volts, impurity@z_0=%f, impurity grad=%f" % (original_bias, original_impurity_z0, originalParams['impurity_grad'] )

    #create a test file for changing the impurity gradient
    configFileSplit = fileName.split(".")
    testConfigFileStr = '.'.join(configFileSplit[:-1]) + "_test." +  configFileSplit[-1]
    os.system(  "cp %s %s" % (fileName, testConfigFileStr) )

    #change the grid size in the copied config file
    replaceConfFileValue(testConfigFileStr, 'xtal_grid', grid_size)

    #should be negative for p type crystal
    realImpurity = original_impurity_z0

    #average impurity in the crystal
    averageImpurity = calcAvgImpurity(crystal_length, realImpurity, impurity_grad)
    print "original average impurity is %f" % averageImpurity

    impurity_z0_lowest = -impurity_grad * crystal_length/10

    if original_impurity_z0 > impurity_z0_lowest:
      print "For this impurity gradient, can't allow impurity_z0 closer to zero than %0.3f" %impurity_z0_lowest
      print "-->Consider adjusting impurity_z0 from %0.3f to %0.3f" % (original_impurity_z0, impurity_z0_lowest)
      exit(0)

    #test the original impurity to make sure it isn't obviously too high
    newDepletion = -1
    while newDepletion == -1:
        newDepletion = findDepletion(testConfigFileStr, original_bias, depletion_tolerance, impurity_z0=original_impurity_z0)
        if newDepletion == -1:
            print "original impurity grad gives depletion is above bias voltage! Adjusting impurity grad down..."
            original_impurity_z0 *= 0.9

    #OK, Let's use this as a starting point.  Check to make sure we aren't already right....
    startDepletion = newDepletion
    deltaDepletion = desiredDepletion - startDepletion
    print "delta depletion is %f" % deltaDepletion

    if abs(deltaDepletion) < depletion_tolerance:
        return original_impurity_z0

    #No? Gonna need to do a golden section search

    #first, guess a new impurity to give us a second bound
    if deltaDepletion > 0:
      #increase concentration to increase depletion voltage (and vice versa)
      #expect change by a slope of ~2500 V/impurityx10^10

      voltageslope =  -2500 / 3 #overestimate by factor of 3 for safety?
      newAvgImpurity = averageImpurity + deltaDepletion / voltageslope
    else:
      #find how low you can possibly go before you're totally out of impurities at the top of the crystal

      newAvgImpurity = calcAvgImpurity(crystal_length, impurity_z0_lowest - 0.001, impurity_grad)

    print "adjusted avg impurity is %f" % newAvgImpurity

#    if newAvgImpurity < 0:
#        print "better come up with a better first guess impurity!"
#        sys.exit()



    newImpurityZ0 = calcImpurityZ0(crystal_length, newAvgImpurity, impurity_grad)

    impurityAtTop = calcImpurityAtEnd(crystal_length, newAvgImpurity, impurity_grad)
    print "Impurity at the top of the crystal is %0.4f" % impurityAtTop
    print "Impurity at the bottom of the crystal is %0.4f" % newImpurityZ0

    if impurityAtTop > 0:
      print "Impurity at the top of the crystal is positive!!"
      exit(0)

#          #return to the original sign
#    newImpurityZ0 = math.copysign(newRealImpurityZ0, original_impurity_z0)



    #test the new guess to make sure its depleted.  Else, screw it, crank the bias voltage (safer than reducing impurity grad)
    newDepletion = -1
    while newDepletion == -1:
        print "new impurity will be " + str(newImpurityZ0)
        newDepletion = findDepletion(testConfigFileStr, original_bias, depletion_tolerance, impurity_z0=newImpurityZ0)
        if newDepletion == -1:
            print "new impurity grad gives depletion is above bias voltage! Increasing bias voltage..."
            original_bias += 250

    goldenCache = {}
    goldenCache[newImpurityZ0] = newDepletion
    goldenCache[original_impurity_z0] = startDepletion

    #Check to make sure the bounds in our search are actually bounding the depletion voltage
    print "Desired depletion is %d, current bounds are %d and %d" % (desiredDepletion, newDepletion, startDepletion)
    if min(newDepletion, startDepletion) <= desiredDepletion <= max(newDepletion, startDepletion):
      print "-->Looks OK!"
    else:
      print "Desired depletion voltage not bounded by starting impurity guesses!  Uh oh"
      sys.exit()

    #implement the golden section search
    #using the abs is "dumb" but easier to code for now

    maxImpurity = max(newImpurityZ0, original_impurity_z0)
    minImpurity = min(newImpurityZ0, original_impurity_z0)
    testMin = maxImpurity - (maxImpurity - minImpurity)/golden
    testMax = minImpurity + (maxImpurity - minImpurity)/golden



    while abs(deltaDepletion) > depletion_tolerance:

        if testMin in goldenCache:
            dep_testMin = goldenCache[testMin]
        else:
            dep_testMin = findDepletion(testConfigFileStr, original_bias, depletion_tolerance, impurity_z0=testMin)

        if testMax in goldenCache:
            dep_testMax = goldenCache[testMax]
        else:
            dep_testMax = findDepletion(testConfigFileStr, original_bias, depletion_tolerance, impurity_z0=testMax)

        #find the minimum of this delta function
        delta_min = abs(desiredDepletion - dep_testMin)
        delta_max = abs(desiredDepletion - dep_testMax)

        if delta_min < delta_max:
            goldenCache[testMin] = dep_testMin
            maxImpurity = testMax
            testMax = testMin
            testMin = maxImpurity - (maxImpurity - minImpurity)/golden
        else:
            goldenCache[testMax] = dep_testMax
            minImpurity = testMin
            testMin = testMax
            testMax = minImpurity + (maxImpurity - minImpurity)/golden

        deltaDepletion = (delta_min + delta_max)/2

    final_impurity = (maxImpurity + minImpurity)/2
    final_depletion = findDepletion(testConfigFileStr, original_bias, depletion_tolerance, impurity_z0=final_impurity)

    print " "
    print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    print "  The new depletion voltage is %d at impurity@z_0=%f" % (final_depletion,final_impurity)
    print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    print " "

    return final_impurity

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

#find the depletion voltage of a given configuration file
def findDepletion(fileName, startingBias, tolerance, impurity_z0="default"):

    print ""
    print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
    print "  Finding depletion voltage for %s at impurity_z0 = " % fileName + str(impurity_z0)
    print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

    depletionString = "is fully depleted"

    if impurity_z0 != "default":
        replaceConfFileValue(fileName, 'impurity_z0', impurity_z0)

    #run fieldgen, capture the output
    fieldGenDir = os.path.expanduser(mjd_siggen_dir)
    args = [fieldGenDir + "mjd_fieldgen", '-c', fileName, "-w", "0", "-p", "0", "-b", str(startingBias)]
    output = subprocess.check_output(args)

    if output.find(depletionString) == -1:
        print "Not depleted at starting bias!"
        return -1


    biasMax = float(startingBias)
    biasMin = 200. #never gonna have a detector deplete under this voltage
    isDepletedMax = True
    isDepletedMin = False

    testMin = biasMax - (biasMax - biasMin)/golden
    testMax = biasMin + (biasMax - biasMin)/golden

    depletionCache = {}

    startTime = time.clock()

    while abs(biasMax - biasMin) > tolerance:
        print "   biasMin is %f, biasMax is %f" % (biasMin, biasMax)

        #calculate max test point (if not already cached)
        if testMax in depletionCache:
            isDepletedMax = depletionCache[testMax]
        else:
            print "      Calculating if depleted at " + str(testMax)
            args[-1] = str(testMax)
            output = subprocess.check_output(args)
            if output.find(depletionString) == -1:
                isDepletedMax = False
            else:
                isDepletedMax = True

        #do the same for the min
        if testMin in depletionCache:
            isDepletedMin = depletionCache[testMin]
        else:
            print "      Calculating if depleted at " + str(testMin)
            args[-1] = str(testMin)
            output = subprocess.check_output(args)
            if output.find(depletionString) == -1:
                isDepletedMin = False
            else:
                isDepletedMin = True

        #check possible search cases
        if isDepletedMax and isDepletedMin:
            depletionCache[testMin] = isDepletedMin

            biasMax = testMax
            testMax = testMin
            testMin = biasMax - (biasMax - biasMin)/golden

        elif not isDepletedMax and not isDepletedMin:
            depletionCache[testMax] = isDepletedMax
            biasMin = testMin
            testMin = testMax
            testMax = biasMin + (biasMax - biasMin)/golden


        elif isDepletedMax and not isDepletedMin:
            #shrink from both sides

            biasMin = testMin
            biasMax = testMax
            testMax = biasMin + (biasMax - biasMin)/golden
            testMin = biasMax - (biasMax - biasMin)/golden

        else:
            print "case not handled!! min is %s, max is %s" % (isDepletedMin, isDepletedMax)
            sys.exit()

    endTime = time.clock()
    elapsedTime = endTime - startTime
    depletionVoltage = (biasMax + biasMin)/2.

    print "Depletion voltage is %f (calulated in %f seconds)" % (depletionVoltage, elapsedTime)

    return (biasMax + biasMin)/2


'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

#find average impurity in a detector in units of e10/cm^4
def calcAvgImpurity(length, impurity_z0, impurity_grad):
    #length is in mm, needs to be in cm

    length /= 10

    return ( impurity_z0 + impurity_grad*length/2)

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

#find impurity at either end given length and gradient (units of e10/cm^4)
def calcImpurityZ0(length, avg_impurity, impurity_grad):
    length /= 10

    return ( avg_impurity - impurity_grad*length/2)

def calcImpurityAtEnd(length, avg_impurity, impurity_grad):
    length /= 10

    return ( avg_impurity + impurity_grad*length/2)

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''


if __name__=="__main__":
    main(sys.argv[1:])
