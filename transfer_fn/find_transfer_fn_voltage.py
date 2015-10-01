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
   Find the transfer function in voltage space by reflecting the input and output waveforms.  Will it work?  Who knows.
   
"""

makePlotsDynamic = 0

desired_length = pow(2,12)
cosFactor = 10;

padding = desired_length - 2*1007
date = (time.strftime("%Y%m%d"))
outFileName = "transferfunction_voltage_%s.root" % date


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


pulserInputWf_orig = MGTWaveform()
pulserInputWf_orig.MakeSimilarTo(pulserInputWf)
pulserInputWf_orig.SetData(pulserInputWf.GetVectorData())


pulserOutputWfData = pulserOutputWf.GetData()
pulserInputWfData = pulserInputWf.GetData()
#

originalLength = pulserOutputWf.GetLength()-1
pulserInputWf.SetLength(2*originalLength)
pulserOutputWf.SetLength(2*originalLength)

amplitudeIn = pulserInputWf[originalLength-1];
amplitudeOut = pulserOutputWf[originalLength-1];

for i in xrange(originalLength):
    pulserInputWf[originalLength + i] = amplitudeIn * math.cos(i*math.pi / (2*originalLength))
    pulserOutputWf[originalLength + i] = amplitudeOut * math.cos(i*math.pi / (2*originalLength))

print "length is %d" % pulserInputWf.GetLength()


pulserInputWf.SetLength(padding+2*originalLength);
pulserOutputWf.SetLength(padding+2*originalLength);

#shifter = MGWFShiftSamples()
#shifter.SetNumberOfShift(padding/2)
#shifter.TransformOutOfPlace(pulserInputWf, test)
#shifter.TransformOutOfPlace(pulserOutputWf, test2)
#
#pulserInputWf = test
#pulserOutputWf = test2


#pulserInputWf.SetLength(desired_length)
#pulserOutputWf.SetLength(desired_length)



if makePlotsDynamic:
    ftHist = pulserInputWf.GimmeHist().Clone()
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
canvas.SetLogy(1)
ftHist.SetTitle("Transfer Function FFT")
ftHist.SetYTitle("AU")
ftHist.Draw()
canvas.Update()
value = raw_input('  --> Press return to continue, q to quit, p to print, l to see previous wf: ')



value = raw_input('  --> Press return to continue, q to quit, p to print, l to see previous wf: ')


analysis_helpers.write_ft_to_file(transferFunctionFT, outFileName, "transfer_function")

#test the convolution



pulserInputWf_orig.SetLength(2**20)



ftHist = pulserInputWf.GimmeHist().Clone()
ftHist.SetTitle("Pulser Function")
ftHist.SetYTitle("AU")
ftHist.Draw()
canvas.Update()
value = raw_input('  --> Press return to continue, q to quit, p to print, l to see previous wf: ')



test5 = MGTWaveformFT()
test5.MakeSimilarTo(transferFunctionFT)

test6 = MGTWaveform()
test6.MakeSimilarTo( pulserInputWf )

fft.PerformFFT(pulserInputWf, test5)

print "input wf: entries %d, period %f, type %s" %(  pulserInputWf.GetLength(), pulserInputWf.GetSamplingPeriod(), pulserInputWf.GetWFTypeName() )
print "transfer ft: entries %d, period %f, type %s" %(  transferFunctionFT.GetTDomainLength(), transferFunctionFT.GetSamplingPeriod(), transferFunctionFT.GetWFTypeName() )

test5 *= transferFunctionFT

print "about to inverse fft"

fft.PerformInverseFFT(test6, test5)

test6.SetLength(originalLength)

ftHist = test6.GimmeHist().Clone()
ftHist.SetTitle("Output Function WF")
ftHist.SetYTitle("AU")
ftHist.Draw()
canvas.Update()
value = raw_input('  --> Press return to continue, q to quit, p to print, l to see previous wf: ')






