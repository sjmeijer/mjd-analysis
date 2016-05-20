#!/usr/bin/python

import sys
import array
import cmath
from ctypes import c_ulonglong
import os
import subprocess
import math
import re

from datetime import datetime

'''
Reads, writes, and replaces siggen conf files

B. Shanks, 5/5/15

'''

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

#read in a configuration file into a map (extend it to read in whatever you need...)
def readConfigurationFile(fileName):
    #find the original parameters we're interested in
    configFile = open(fileName , "r")
    
    bias = 0.
    impurity_grad = 0.
    impurity_z0 = 0.
    xtal_length = 0.
    
    for line in configFile:
        if findParameter("xtal_HV", line) is not None:
            bias = readParameter(line)
        elif findParameter("impurity_z0", line) is not None:
            impurity_z0 = readParameter(line)
        elif findParameter("impurity_gradient", line) is not None:
            impurity_grad = readParameter(line)
        elif findParameter("xtal_length", line) is not None:
            xtal_length = readParameter(line)

    configFile.close()
    
    configMap = {'bias':bias,
                'impurity_grad':impurity_grad,
                'impurity_z0':impurity_z0,
                'xtal_length':xtal_length}

    return configMap

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

def findParameter(paramName, searchString):
    #regex on what i want to replace - parameter name, whitespace, then a number (optional minus sign, optional decimal point)
    #skips commented lines
    #i guess right now it really only works if the value is used once, but if you define it twice without comments, what are you doing anyway?
    regex = "(?<!#)" + paramName + "\s+" + "-?[0-9]\d*(\.\d+)?"
    pattern = re.compile(regex)
    m = pattern.search(searchString)

    if m is None:
        return None
    else:
        return m.group()


'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

def findStringParameter(paramName, searchString):
    #regex on what i want to replace - parameter name, whitespace, then a number (optional minus sign, optional decimal point)
    #skips commented lines
    #i guess right now it really only works if the value is used once, but if you define it twice without comments, what are you doing anyway?
    regex = "(?<!#)" + paramName + "\s+" + ".+"
    pattern = re.compile(regex)
    m = pattern.search(searchString)

    if m is None:
        return None
    else:
        return m.group()
    


'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

#replace a number in a line of a siggen conf file
def replaceConfFileValue(fileName, paramName, newValue):
    
    inFile = open(fileName, 'r')
    fileContent = inFile.read()
    inFile.close()
    
    if not isinstance(newValue, basestring):
      print "replacing %s to %s in %s" % (paramName, str(newValue), fileName)
      newContent = replaceValue(fileContent, paramName, newValue)
    else:
      print "replacing %s to %s in %s" % (paramName, newValue, fileName)
      newContent = replaceStringValue(fileContent, paramName,newValue)
    
    outFile = open(fileName, 'w')
    outFile.write(newContent)
    outFile.close()

#replaces a siggen conf value in given string
def replaceValue(replaceString, paramName, newValue):
    
    oldLine = findParameter(paramName, replaceString)
    
    if oldLine is None:
        print "No parameter with name %s was found!" % (paramName)
        sys.exit()
        return None

    newLine = re.sub("\s[-+]?\d+[\.]?\d*", " " + str(newValue), oldLine)

    print "     " + oldLine
    print "     " + newLine + " (new value is %f)" % newValue
    
    newString = replaceString.replace(oldLine, newLine)

    return newString

#replaces a siggen conf value in given string
def replaceStringValue(replaceString, paramName, newValue):
    
    oldLine = findStringParameter(paramName, replaceString)
    
    if oldLine is None:
        return None
        print "No parameter with name %s was found!" % (paramName)
        sys.exit
    old_split = oldLine.split(None, 2)
    old_split[1] = str(newValue)

    newLine = old_split[0]
    for bit in old_split[1:]:
        newLine += "    " + bit

    newString = replaceString.replace(oldLine, newLine)
    return newString


'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

#find a number on a line of a siggen config file immediately after a given string
# ie, it reads in
# not very smart, but should work for any line that passes the regex
def readParameter(line):
    
    print line
    
    #strip white space and newlines (probably not necessary)
    line = line.rstrip()
    
    #pull off any trailing comments (totally unnecessary)
    line = line.split("#")[0]
    
    #split it by spaces, take the second value
    valueStr = line.split()[1]
    
    #cast it to a float
    value = 0.
    try:
        value = float(valueStr)
    except ValueError:
        print "Couldn't find a number value in " + line
        sys.exit()
    
    return value

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''
#class for holding siggen conf dat
class SiggenCrystalInfo:
    
    kCrystalTypes = ['ortec', 'bege'] #valid crystal types

    def __init__(self):
        self.fCrystalType = None
        self.fCrystalID = None
        self.fDepletionVoltage = None
        
        #properties which will get written to file
        self.siggen_xtal_radius = None
        self.siggen_xtal_length = None
        self.siggen_pc_radius = None
        self.siggen_pc_length = None
        self.siggen_Li_thickness = None
        self.siggen_impurity_z0 = None
        self.siggen_impurity_gradient = None
        self.siggen_xtal_HV = None
    
    def __iter__(self):
        for attr, value in self.__dict__.iteritems():
            yield attr, value

    # requires a template file to replace on
    def WriteToConfFile(self, copyFile):
        #pull in file content
        inFile = open(copyFile, 'r')
        fileContent = inFile.read()
        inFile.close()
    
        #add a stamp to the start of the file
        stamp = "#Siggen conf file for %s.  Autogenerated at %s\n\n" % (self.fCrystalID, str(datetime.now()) )
        fileContent = stamp + fileContent
    
        #replace the right parameters
        paramDict = dict(self)
    
        for field in paramDict:
            if field.startswith("siggen_") and paramDict[field] is not None:
                fieldName = field[7:]
                fileContent = replaceValue(fileContent, fieldName, paramDict[field])
                print "Replaced %s with %f" % (fieldName, paramDict[field])
    
        #also change the field and wp names
        fileContent = replaceStringValue(fileContent, 'field_name', "fields/%s_ev.dat" % self.fCrystalID)
        fileContent = replaceStringValue(fileContent, 'wp_name', "fields/%s_wp.dat" % self.fCrystalID)

        #write the thing
        outFileName = "%s_autogen.conf" % self.fCrystalID
        outFile = open(outFileName, 'w')
        outFile.write(fileContent)
        outFile.close()

        return outFileName

    def SetCrystalType(self, crystalType):
        if crystalType in self.kCrystalTypes:
            self.fCrystalType = crystalType
        else:
            raise ValueError("Crystal type %f is not a valid siggen crystal type" % crystalType)

