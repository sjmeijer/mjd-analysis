#!/usr/bin/python
import ROOT
from ROOT import *
TROOT.gApplication.ExecuteFile("$MGDODIR/Root/LoadMGDOClasses.C")
TROOT.gApplication.ExecuteFile("$MGDODIR/Majorana/LoadMGDOMJClasses.C")
TROOT.gApplication.ExecuteFile("$GATDIR/LoadGATClasses.C")
import sys
import array
import cmath
from ctypes import c_ulonglong
import os
import math

import scipy as sp
import numpy
import time

sys.path.append("/Users/bshanks/Dev/Analysis/")
import analysis_helpers


"""
   For a set of runs, make a pulser superpulse
   
   uses cuts on energy and A/E, which require manual input
   
"""

makePlotsDynamic = 0

desired_length = 2**12

padding = desired_length - 1008
date = (time.strftime("%Y%m%d"))
outFileName = "transferfunction_current_%s.root" % date


canvas = TCanvas("canvas", "",700,450)
pad2 = TPad("pad2","The Transfer function?", 0.05,0.05,0.95,0.95)
gStyle.SetOptStat(0)
pad2.Draw()
pad2.cd()


pulserInputFileName = "Pulser_in_superpulse_20150326.root"
pulserOutputFileName = "Pulser_out_superpulse_20150326.root"

pulserInputFile = TFile(pulserInputFileName)
pulserOutputFile = TFile(pulserOutputFileName)

#pulserInputFile.ls()
#pulserOutputFile.ls()

pulserInputTree = pulserInputFile.Get("pulser_superpulse")
pulserOutputTree = pulserOutputFile.Get("pulser_superpulse")

#pulserInputTree.Print()
#pulserOutputTree.Print()


pulserInputBranch  = pulserInputTree.GetBranch("PulserWaveform")
pulserOutputBranch  = pulserOutputTree.GetBranch("PulserWaveform")

pulserInputWf = MGTWaveform();
pulserInputBranch.SetAddress(AddressOf(pulserInputWf))
pulserInputBranch.GetEntry(0)

pulserOutputWf = MGTWaveform();
pulserOutputBranch.SetAddress(AddressOf(pulserOutputWf))
pulserOutputBranch.GetEntry(0)

print "pulser input length is %d" % pulserInputWf.GetLength()
print "pulser output length is %d" % pulserOutputWf.GetLength()

derivIn = MGTWaveform();
derivOut = MGTWaveform();

deriv = MGWFDerivative()

pulserInputWf_orig = pulserInputWf

deriv.TransformOutOfPlace(pulserInputWf, derivIn)
pulserInputWf = derivIn
deriv.TransformOutOfPlace(pulserOutputWf, derivOut)
pulserOutputWf = derivOut



#pulserOutputWfData = pulserOutputWf.GetData()
#pulserInputWfData = pulserInputWf.GetData()
#


#for i in xrange(padding):
#    pulserInputWf[i] = 0
#    pulserOutputWf[i] = 0
#    pulserInputWf[padding + originalLength + i] = 0
#    pulserOutputWf[padding + originalLength+ i] = 0
#
#for i in xrange(originalLength):
#    pulserInputWf[padding + i] = pulserInputWfData[i]
#    pulserOutputWf[padding + i] = pulserOutputWfData[i]

print "length is %d" % pulserInputWf.GetLength()

originalLength = pulserOutputWf.GetLength()
pulserInputWf.SetLength(padding+originalLength);
pulserOutputWf.SetLength(padding+originalLength);
test = MGTWaveform();
test2 = MGTWaveform();

#shifter = MGWFShiftSamples()
#shifter.SetNumberOfShift(padding/2)
#shifter.TransformOutOfPlace(pulserInputWf, test)
#shifter.TransformOutOfPlace(pulserOutputWf, test2)
#
#pulserInputWf = test
#pulserOutputWf = test2


pulserInputWf.SetLength(desired_length)
pulserOutputWf.SetLength(desired_length)



if makePlotsDynamic:
    ftHist = test.GimmeHist().Clone()
    ftHist.SetTitle("Input Function")
    ftHist.SetYTitle("AU")
    ftHist.Draw()
    canvas.Update()
    value = raw_input('  --> Press return to continue, q to quit, p to print, l to see previous wf: ')

if makePlotsDynamic:
    ftHist = pulserOutputWf.GimmeHist().Clone()
    ftHist.SetTitle("Output Function")
    ftHist.SetYTitle("AU")
    ftHist.Draw()
    canvas.Update()
    value = raw_input('  --> Press return to continue, q to quit, p to print, l to see previous wf: ')




pulserInputFFT = MGTWaveformFT()
pulserOutputFFT = MGTWaveformFT()


fft = MGTWFFastFourierTransformFFTW()
#fft.SetNormalization(MGTWFFastFourierTransformDefault.kInverse)
fft.PerformFFT(pulserInputWf, pulserInputFFT)
fft.PerformFFT(pulserOutputWf, pulserOutputFFT)

if makePlotsDynamic:
    ftHist = pulserInputFFT.GimmeHist().Clone()
    ftHist.SetTitle("Input Function FFT")
    ftHist.SetYTitle("AU")
    ftHist.Draw()
    canvas.Update()
    value = raw_input('  --> Press return to continue, q to quit, p to print, l to see previous wf: ')

if makePlotsDynamic:
    ftHist = pulserOutputFFT.GimmeHist().Clone()
    ftHist.SetTitle("Output Function FFT")
    ftHist.SetYTitle("AU")
    ftHist.Draw()
    canvas.Update()
    value = raw_input('  --> Press return to continue, q to quit, p to print, l to see previous wf: ')




transferFunctionWf = MGTWaveform()
transferFunctionWf.MakeSimilarTo(pulserOutputWf)
transferFunctionFT = pulserOutputFFT




transferFunctionFT/=pulserInputFFT



#if makePlotsDynamic:
ftHist = transferFunctionFT.GimmeHist().Clone()
ftHist.SetTitle("Transfer Function FFT")
ftHist.SetYTitle("AU")
ftHist.Draw()
canvas.Update()
value = raw_input('  --> Press return to continue, q to quit, p to print, l to see previous wf: ')

analysis_helpers.write_ft_to_file(transferFunctionFT, outFileName, "transfer_function")

int = MGWFIntegral()
integratedWF = MGTWaveform()

#test the convolution

ftHist = pulserInputWf.GimmeHist().Clone()
ftHist.SetTitle("Pulser Function")
ftHist.SetYTitle("AU")
ftHist.Draw()
canvas.Update()
value = raw_input('  --> Press return to continue, q to quit, p to print, l to see previous wf: ')


test4 = MGTWaveformFT()
test5 = MGTWaveform()
test6 = MGTWaveform()

fft.PerformFFT(pulserInputWf, test4)

test4 *= transferFunctionFT

fft.PerformInverseFFT(test5, test4)

test5.SetLength(originalLength)

int.TransformOutOfPlace(test5, integratedWF)

ftHist = integratedWF.GimmeHist().Clone()
ftHist.SetTitle("Output Function FFT")
ftHist.SetYTitle("AU")
ftHist.Draw()
canvas.Update()
value = raw_input('  --> Press return to continue, q to quit, p to print, l to see previous wf: ')






