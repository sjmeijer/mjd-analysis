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


"""
   For a set of runs, make a pulser superpulse
   
   uses cuts on energy and A/E, which require manual input
   
"""

makePlotsDynamic = 0

desired_length = 2**20

padding = desired_length - 1008
date = (time.strftime("%Y%m%d"))
outFileName = "transferfunction_%s.root" % date


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

shifter = MGWFShiftSamples()
shifter.SetNumberOfShift(padding/2)
shifter.TransformOutOfPlace(pulserInputWf, test)
shifter.TransformOutOfPlace(pulserOutputWf, test2)

pulserInputWf = test
pulserOutputWf = test2


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





canvas3 = TCanvas("canvas3", "",700,450)
pad3 = TPad("pad3","Transfer Function FFTs", 0.05,0.05,0.95,0.95)
gStyle.SetOptStat(0)
pad3.Draw()
pad3.cd()


hist2 = pulserInputFFT.GimmeHist().Clone()
hist2.SetTitle("Transfer Function FFTs")
hist2.SetYTitle("AU")
hist2.SetLineColor(kBlack)
hist2.Draw()

hist3 = pulserOutputFFT.GimmeHist().Clone()
hist3.SetTitle("Transfer Function FFTs")
hist3.SetYTitle("AU")
hist3.SetLineColor(kRed)
hist3.Draw("SAME")

transferFunctionFT/=pulserInputFFT

hist1 = transferFunctionFT.GimmeHist().Clone()
hist1.SetTitle("Transfer Function FFTs")
hist1.SetYTitle("AU")
hist1.Draw("SAME")


leg2 = TLegend(0.525,0.7,0.82,0.85)
leg2.SetTextFont(72);
leg2.AddEntry(hist1,"Transfer Function","l")
leg2.AddEntry(hist2,"Pulser Input","l")
leg2.AddEntry(hist3,"Pulser Output","l")

leg2.Draw();

pad3.SetLogy(1)
canvas3.Update()
value = raw_input('  --> Press return to continue, q to quit, p to print, l to see previous wf: ')

if makePlotsDynamic:
    ftHist = transferFunctionFT.GimmeHist().Clone()
    ftHist.SetTitle("Transfer Function FFT")
    ftHist.SetYTitle("AU")
    ftHist.Draw()
    canvas.Update()
    value = raw_input('  --> Press return to continue, q to quit, p to print, l to see previous wf: ')



fft.PerformInverseFFT(transferFunctionWf, pulserOutputFFT)

#shift it back
shifter.SetNumberOfShift(-padding/2)

test3 = MGTWaveform()
shifter.TransformOutOfPlace(transferFunctionWf, test3)
test3.SetLength(originalLength);
transferFunctionWf = test3

print "The transfer function has length %d" % transferFunctionWf.GetLength()
#transferFunctionWf[0] = transferFunctionWf[1]

##normalize it
counter = 0;
for i in xrange(transferFunctionWf.GetLength() ):
    counter += transferFunctionWf[i]
transferFunctionWf/=counter

ftHist = transferFunctionWf.GimmeHist().Clone()
ftHist.SetTitle("Transfer Function?")
ftHist.SetYTitle("AU")
ftHist.Draw()

canvas.Update()

value = raw_input('  --> Press return to continue, q to quit, p to print, l to see previous wf: ')


outfile = TFile(outFileName, "RECREATE");
outTree = TTree("transfer_function", "Transfer Function from Pulser Data")
outTree.Branch("TransferFunctionWaveform", "MGTWaveform", transferFunctionWf);

outTree.Fill()
outfile.Write()
outfile.Close()







