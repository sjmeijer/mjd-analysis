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

from scipy import ndimage
import numpy as np
import mallat_wavelets


"""
   For a set of runs, calibrate off given peaks, with given guesses
   
"""

dataSetName = "surfprot"
detectorName = "P3DCR"
gatDataName = "mjd_run"
gatTreeName = "mjdTree"

builtDataName = "OR_run"
builtTreeName = "MGTree"


runRange = (35001701,35001701)

nvoice = 12.;
oct = 5.
scale = 64.



def main():
    for iRun in range( runRange[0],  runRange[1]+1):
        print 'processing run', iRun
        gatFilePath = "$MJDDATADIR/" + dataSetName + "/data/gatified/" + detectorName + "/" + gatDataName + "%d.root" % iRun
        gat_file = TFile(gatFilePath)
        gat_tree = gat_file.Get(gatTreeName)
        
        builtFilePath = "$MJDDATADIR/" + dataSetName + "/data/built/" + detectorName + "/" + builtDataName + "%d.root" % iRun
        built_file = TFile(builtFilePath)
        built_tree = built_file.Get(builtTreeName)

        makeSkimFile(built_tree, gat_tree)


#makeSkimFile(chain, calibrationMap)

def calcWaveletParam(data, nvoice, oct, scale):
    
    wavelet = 'DerGauss';
    rwt = mallat_wavelets.rwt(data,nvoice,wavelet,oct,scale)
    
    #calcs
    rwt = np.flipud( np.transpose(rwt) )
    (nscale,n) = rwt.shape;
    
    max_distances = np.empty(n)
    max_distances[:]=3
    gap_thresh = 3
    
    ridges = mallat_wavelets.scipy_ridges(rwt, max_distances, gap_thresh)
    #filteredRidges = mallat_wavelets.filter_ridge_lines(rwt, ridges, min_length = 10)
    
    waveletParamMax = np.zeros(3)
    waveletParamSecondMax = np.zeros(3)

    
    for ridge in ridges:
        waveletParam = np.amax(ridge[0][-1] *rwt[ ridge[0][-1],  ridge[1][-1]])
#        waveletParamIdx = np.argmax(ridge[0] *rwt[ ridge[0],  ridge[1]])
#        print "idx: %d of %d" % (waveletParamIdx, len(ridge[0]) -1)
        if waveletParam > waveletParamSecondMax[0]:
            if waveletParam > waveletParamMax[0]:
                waveletParamSecondMax[:] = waveletParamMax[:]
                waveletParamMax = [waveletParam, ridge[0][-1], rwt[ ridge[0][-1],  ridge[1][-1]] ]
            else:
                waveletParamSecondMax = [waveletParam, ridge[0][-1], rwt[ ridge[0][-1],  ridge[1][-1]] ]

#    print "max: " + str(waveletParamMax)
#    print "second max" + str(waveletParamSecondMax)
##
#    value = raw_input('  --> Press q to quit, any other key to continue\n')
#    exit(1)


    return waveletParamSecondMax[0]

##############################################################################################
def makeSkimFile(builtTree, gatTree):
    
    
    gaussianFilterSigma = 3
    
    builtTree.AddFriend(gatTree)

    numEntries = builtTree.GetEntries()
    for iEntry in xrange(numEntries):
        update_progress(float(iEntry)/numEntries)
        
        builtTree.GetEntry(iEntry)
        
        gatTree.GetEntry(iEntry)
        
        
        
        event = builtTree.event
        
        for i_wfm in xrange( event.GetNWaveforms() ):
            digData   = event.GetDigitizerData( i_wfm )
            channel   = digData.GetChannel()
        
            wf = event.GetWaveform(i_wfm)

            orig_length = wf.GetLength()
            next_pow_2 = np.ceil(np.log2(orig_length))
            #print "next pow 2 is " + str(next_pow_2)
            wflength = int(pow(2,next_pow_2))

            wf.SetLength(wflength)

            #reflective extension to 2^n
            for i in xrange(wflength-orig_length):
                at_sizet = vector("size_t")(1)
                at_sizet[0] = int((orig_length-i)-1)
                #print orig_length-i
                wf.SetValue(i+orig_length, wf.At(at_sizet[0]))
            
            np_data = wf.GetVectorData()
            np_data_smoothed = ndimage.filters.gaussian_filter1d(np_data, gaussianFilterSigma)
            
            wfParam = calcWaveletParam(np_data_smoothed, nvoice, oct, scale)

#            print wfParam
#            value = raw_input('  --> Press q to quit, any other key to continue\n')
#            exit(1)


#print "wf param: %f" % wfParam



#    outfilename = "wavelet_skim.root";
#    oFile = TFile(outfilename,"recreate");
#    oFile.cd();
#    outTree = TTree("waveletTree","Only energy");
#
#    EnergykeV = vector("double")()
#    EnergyBoard = vector("double")()
#    EnergyRaw = vector("double")()
#    Chan = vector("double")()
#    
#    inRun = array.array( 'd', [ 0 ] )
#    
#    outTree.Branch("EnergykeV",EnergykeV);
#    outTree.Branch("EnergyBoard",EnergyBoard);
#    outTree.Branch("EnergyRaw",EnergyRaw);
#    outTree.Branch("channel",Chan);
#    outTree.Branch("run",inRun, "run\D");
#
#    inEnergyRaw = []
#
#    for iChan in xrange(len(channels)):
#        if (energyName == "energy"):
#            inEnergyRaw.append("energy");
#        else:
#            print "Energies besides onboard not yet supported.  I'm sorta lazy."
#
#    print "there are %d entries in the chain " % chain.GetEntries()
#    for iEntry in xrange(chain.GetEntries()):
#        update_progress(iEntry/float(chain.GetEntries()))
#        
#        EnergykeV.clear();
#        EnergyRaw.clear();
#        EnergyBoard.clear();
#        Chan.clear();
#        
#        #chain.LoadTree( iEntry )
#        chain.GetEntry(iEntry)
#        
#        inEnergyBoard = chain.energy
#        inChan = chain.channel
#        inRun = chain.run
#        
#        chanLength = inChan.size()
#        
#        #print "chanLength is " + str(chanLength)
#
#        EnergykeV.resize(chanLength);
#        EnergyRaw.resize(chanLength);
#        EnergyBoard.resize(chanLength);
#        Chan.resize(chanLength);
#
#        for ichan in xrange(chanLength):
#            dchan = inChan.at(ichan)
#            
#            Chan[ichan]=dchan;
#            eboard=inEnergyBoard.at(ichan);
#            eraw=inEnergyBoard.at(ichan);
#
#            EnergyRaw[ichan]= eraw;
#            EnergyBoard[ichan] = eboard;
#            
#            chan_sizet = vector("size_t")(1)
#            chan_sizet[0] = int(dchan)
#            
#            runNumber = vector("size_t")(1)
#            runNumber[0] = int(inRun)
#            
#            scale = calMap.GetScale(chan_sizet.at(0), energyName, runNumber.at(0))
#            offset = calMap.GetOffset(chan_sizet.at(0), energyName, runNumber.at(0))
#            
#            EnergykeV[ichan] = offset + scale*EnergyRaw[ichan]
#        
##            print "Runnumber is " + str(runNumber.at(0))
#
##            print "chan is " + str(ichan)
##            print "energyname is " + str(energyName)
##            print "scale is " + str( scale)
##            print "offset is " + str( offset)
##            print "energyKeV is " + str( EnergykeV[ichan])
#
#        outTree.Fill();
#    print ""
#    print "out tree has %d entries" % outTree.GetEntries();
#    oFile.cd();
#    oFile.Write();
#
##oFile.Write("", kOverwrite);
#    oFile.Close();

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


