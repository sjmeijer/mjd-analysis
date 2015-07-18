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
import re

import scipy as sp
import numpy


"""
   For a set of runs, calibrate off given peaks, with given guesses
   
"""

#channel = 66
channels = [66,68]
runRange = (35001701,35001788)# 35001767)
#runRange = (35001701,35001711)# 35001767)

calPeakskeV = vector('double')(4)
calPeakskeV[0] = 583.187 #tl208
calPeakskeV[1] = 1460.822 #k40
calPeakskeV[2] = 1764.494 #bi214
calPeakskeV[3] = 2614.7 #tl208

calPeakskeV = vector('double')(4)
calPeakskeV[0] = 583.187
calPeakskeV[1] = 1460.822
calPeakskeV[2] = 1764.494
calPeakskeV[3] = 2614.7

hiGainGuesses = vector('double')(4)
hiGainGuesses[0] = 205 * 10**3
hiGainGuesses[1] = 515 * 10**3
hiGainGuesses[2] = 622 * 10**3
hiGainGuesses[3] = 921 * 10**3

loGainGuesses = vector('double')(4)
loGainGuesses[0] = 61 * 10**3
loGainGuesses[1] = 153 * 10**3
loGainGuesses[2] = 186 * 10**3
loGainGuesses[3] = 275 * 10**3


peakGuessMap = {66: hiGainGuesses, #higain
                68: loGainGuesses} #lo gain

dataSetName = "surfstr"
detectorName = "P3DCR"
gatDataName = "mjd_run"

energyName = "energy"

peakWidths = vector("size_t")(4)
peakWidths[0] = 2
peakWidths[1] = 2
peakWidths[2] = 2
peakWidths[3] = 2

def main():
    chain = TChain("mjdTree")
    chain.SetDirectory(0)

    # add files to the chain:
    for iRun in range( runRange[0],  runRange[1]+1):
        print 'processing run', iRun
        chain.Add( "$MJDDATADIR/" + dataSetName + "/data/gatified/" + detectorName + "/" + gatDataName + "%d.root" % iRun)
#    
    calibrationMap = GATCalibrationMap()
    calibrateSpec = GATCalibrateSpectrum(calPeakskeV)
    calibrateSpec.SetPeakFitRange(2.0); #no clue why 2
    calibrateSpec.UseADCGuess();

    for chan, guess in peakGuessMap.iteritems():
        calibrateADC(chain, chan, guess, calibrateSpec, calibrationMap)

    outcalibrationfile = "calibration_byrun.dat";
    calibrationMap.WriteCalibrationMapToTextFile(outcalibrationfile);

    outcalibrationfile = "calibration_byrun.dat";
    calibrationMap = GATCalibrationMap()
    calibrationMap.ReadCalibrationMapFromTextFile(outcalibrationfile);
    
    for iRun in range( runRange[0],  runRange[1]+1):
      reWriteEnergyCal( "/Users/bshanks/Data/" + dataSetName + "/data/gatified/" + detectorName + "/" + gatDataName + "%d.root" % iRun, calibrationMap)

#makeSkimFile(chain, calibrationMap)

##############################################################################################

def calibrateADC(chain, channel, adcGuess, gatCalibrator, gatCalMap):

    gatCalibrator.SetPeakADCGuess(adcGuess, peakWidths);
    
    chanCut = TCut("channel == " + str(channel))
    
    adcHist = TH1F("hADC_"+ str(channel),"Uncalibrated",2000,0,adcGuess[3]*1.1);
    chain.Project("hADC_"+str(channel), energyName, "channel == " + str(channel));
    canvas = TCanvas("canvas", "",700,450)
    pad = TPad("pad","Trap Filters", 0.05,0.05,0.95,0.95)
    gStyle.SetOptStat(0)
    pad.Draw()
    pad.cd()
    pad.SetLogy()
    chain.Project("hADC_"+str(channel), energyName, "channel == " + str(channel));
    adcHist.Draw()
    canvas.Update()
    value = raw_input('  --> Make sure the binning is OK ')
    
    gatCalibrator.SavePeakFits(True,"Calibrated_channel"+str(channel))

    gatCalibrator.CalculateCalibrationConstants(adcHist);
    
    chan_sizet = vector("size_t")(1)
    chan_sizet[0] = int( channel )
            
    runNumber = vector("size_t")(1)
    runNumber[0] = int( runRange[0] )
                        
    scale = gatCalibrator.GetScale(  )
    offset = gatCalibrator.GetOffset(  )
                                   

    value = raw_input('  --> Make sure the binning is OK ')

    energyHist = TH1F("hCal_"+ str(channel),"Calibrated",2700,0,2700);
    chain.Project("hCal_"+str(channel), " %s * %f + %f" % (energyName, scale, offset), "channel == " + str(channel));
    energyHist.Draw()
    canvas.Update()                                                    
    value = raw_input('  --> Make sure the calib is OK ')

    for run in range(runRange[0], runRange[1]+1):
        gatCalibrator.FillCalibrationMap(gatCalMap, channel,energyName,run,run);

def reWriteEnergyCal(fileName, calMap):
    if not (os.path.isfile(fileName)):
      print "Skipping " + fileName
      return
  
    #parse for the run number
    fileparts = re.split("run",fileName)
    fileparts2 = re.split("\.",fileparts[1]);
    run= fileparts2[0]
    
    print "on run number %s" % run
    
    runNumber = vector("size_t")(1)
    runNumber[0] = int(run)
    
    newfileName = fileparts[0] + "cal_run" + fileparts[1]
    
    fFile = TFile.Open(fileName,"read")
    fTree = TTree();
    fFile.GetObject("mjdTree", fTree);

    fTree.SetBranchStatus("energyCal", 0);

    energyADC = vector("double")()
    energyCal = vector("double")()
    
    out_file = TFile(newfileName,"recreate")
    outtree = fTree.CloneTree(0)
    
    energyName = "energy"
    
    fTree.SetBranchAddress(energyName,energyADC);
    outtree.Branch("energyCal",energyCal);
    
    for ientry in xrange( fTree.GetEntries() ):
      fTree.GetEntry(ientry)
      
      numChans = fTree.channel.size()
      energyCal.resize( numChans )
      
      for iChan in xrange( numChans ):
        
        chan_sizet = vector("size_t")(1)
        chan_sizet[0] = int( fTree.channel.at(iChan) )
        
        runNumber = vector("size_t")(1)
        runNumber[0] = int(run)
        scale = calMap.GetScale( chan_sizet.at(0) , energyName, runNumber.at(0) )
        offset = calMap.GetOffset( chan_sizet.at(0), energyName, runNumber.at(0) )
        
        energyCal[iChan] = offset + scale*energyADC.at(iChan)
      # if energyCal[iChan] > 2500:
      #print "energyCal %f at channel %d in run %d with energy name %s, scale %f, offset %f" % (energyCal.at(iChan), int( chan_sizet.at(0) ), runNumber.at(0), energyName, scale, offset)
      outtree.Fill();
    
    outtree.Write()
    
    out_file.Close()
    fFile.Close()


##############################################################################################
def makeSkimFile(chain, calMap):

    outfilename = "energyskim.root";
    oFile = TFile(outfilename,"recreate");
    oFile.cd();
    outTree = TTree("energyTree","Only energy");

    EnergykeV = vector("double")()
    EnergyBoard = vector("double")()
    EnergyRaw = vector("double")()
    Chan = vector("double")()
    
    inRun = array.array( 'd', [ 0 ] )
    
    outTree.Branch("EnergykeV",EnergykeV);
    outTree.Branch("EnergyBoard",EnergyBoard);
    outTree.Branch("EnergyRaw",EnergyRaw);
    outTree.Branch("channel",Chan);
    outTree.Branch("run",inRun, "run\D");

    inEnergyRaw = []

    for iChan in xrange(len(channels)):
        if (energyName == "energy"):
            inEnergyRaw.append("energy");
        else:
            print "Energies besides onboard not yet supported.  I'm sorta lazy."

    print "there are %d entries in the chain " % chain.GetEntries()
    for iEntry in xrange(chain.GetEntries()):
        update_progress(iEntry/float(chain.GetEntries()))
        
        EnergykeV.clear();
        EnergyRaw.clear();
        EnergyBoard.clear();
        Chan.clear();
        
        #chain.LoadTree( iEntry )
        chain.GetEntry(iEntry)
        
        inEnergyBoard = chain.energy
        inChan = chain.channel
        inRun = chain.run
        
        chanLength = inChan.size()
        
        #print "chanLength is " + str(chanLength)

        EnergykeV.resize(chanLength);
        EnergyRaw.resize(chanLength);
        EnergyBoard.resize(chanLength);
        Chan.resize(chanLength);

        for ichan in xrange(chanLength):
            dchan = inChan.at(ichan)
            
            Chan[ichan]=dchan;
            eboard=inEnergyBoard.at(ichan);
            eraw=inEnergyBoard.at(ichan);

            EnergyRaw[ichan]= eraw;
            EnergyBoard[ichan] = eboard;
            
            chan_sizet = vector("size_t")(1)
            chan_sizet[0] = int(dchan)
            
            runNumber = vector("size_t")(1)
            runNumber[0] = int(inRun)
            
            scale = calMap.GetScale(chan_sizet.at(0), energyName, runNumber.at(0))
            offset = calMap.GetOffset(chan_sizet.at(0), energyName, runNumber.at(0))
            
            EnergykeV[ichan] = offset + scale*EnergyRaw[ichan]
        
#            print "Runnumber is " + str(runNumber.at(0))

#            print "chan is " + str(ichan)
#            print "energyname is " + str(energyName)
#            print "scale is " + str( scale)
#            print "offset is " + str( offset)
#            print "energyKeV is " + str( EnergykeV[ichan])

        outTree.Fill();
    print ""
    print "out tree has %d entries" % outTree.GetEntries();
    oFile.cd();
    oFile.Write();

#oFile.Write("", kOverwrite);
    oFile.Close();

#I got impatient with loading longer runs, so this prints the progress.
def update_progress(progress):
    barLength = 10 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), round(progress*100,2) , status)
    sys.stdout.write(text)
    sys.stdout.flush()


if __name__=="__main__":
    main()


