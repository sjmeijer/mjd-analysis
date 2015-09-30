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
   
   requires Micah's new superpulse code
   
"""
### dd/mm/yyyy format
date = (time.strftime("%Y%m%d"))

#for the "output" pulser thru the electronics chain

channel = 68
outFileName = "Pulser_out_superpulse_%s" % date
runRange = (35001658, 35001658)
pulserEnergy = 136.5*pow(10,3) #defines center of energy cut
pulserRes = .75*pow(10,3) #defines +/- around pulser energy to cut

#channel = 70
#outFileName = "Pulser_in_superpulse_%s" % date
#runRange = (35001657, 35001657)
#pulserEnergy = 909.25*pow(10,3) #defines center of energy cut
#pulserRes = .25*pow(10,3) #defines +/- around pulser energy to cut

numPulses = 1000


boardChannel = channel  & int("00001111",2)
print "board channel is %d" %boardChannel


def main():

    # Define your cuts
    energyCut = "energy>%f && energy<%f" % (pulserEnergy-pulserRes, pulserEnergy+pulserRes);
    channelCut = "channel == " + str(channel)
    aeCut = " abs(rcDeriv50nsMax/energy) > .00054 && abs(rcDeriv50nsMax/energy) < .00062  "
    cut = energyCut + " && "+ channelCut #+ " && " + aeCut
    print "The cuts will be: " + cut


    #Prepare the MGDO classes we will use

    #flat-time will be used for a baseline subtraction
    flatTime = 300;
    #Instantiate and prepare the baseline remover transform:
    baseline = MGWFBaselineRemover()
    baseline.SetBaselineTime(flatTime)

    #Instantiate ChiSquareCalculator
    chiSquare = MGWFCalculateChiSquare()

    #Time point calculator for 10-90 risetime
    timePointCalc = MGWFTimePointCalculator();
    timePointCalc.AddPoint(.1); #find the 10% time point
    timePointCalc.AddPoint(.9); #find the 90% time point


    fitter = MGWFFitter();
    sp = MGTSuperpulse();
    fitter.SetShiftMeth(MGWFFitter.kChiShift);
    fitter.SetNormMeth(MGWFFitter.kChiNorm);
    
    isTemplateSet = 0

    chain = TChain("MGTree")
    chain.SetDirectory(0)
    chain2 = TChain("mjdTree")
    chain2.SetDirectory(0)        


    iRange=0
# add files to the chain:
    for iRun in range( runRange[0],  runRange[1]+1):
        print 'processing run', iRun
        chain.Add( "$MJDDATADIR/surfprot/data/built/P3DCR/OR_run%d.root" % iRun)
        chain2.Add( "$MJDDATADIR/surfprot/data/gatified/P3DCR/mjd_run%d.root" % iRun)
    
    chain.AddFriend(chain2,"mjdTree")
    chain2.SetEntryList(0)
    chain2.Draw(">>elist%d" % iRange, cut, "entrylist%d" % iRange)
    elist = gDirectory.Get("elist%d" % iRange)
    print "Number of entries in the entryList is " + str(elist.GetN())
    chain2.SetEntryList(elist);

    listEntries = elist.GetN()
    #chainEntries = chain.getEntries();

    startTime = 0;
    wfmCounter = 0;

    #Loop over entries in the chain
    for ientry in xrange( listEntries):
        entryNumber = chain2.GetEntryNumber(ientry);
        chain.LoadTree( entryNumber )
        chain.GetEntry( entryNumber )
        update_progress(ientry/float(listEntries))
        # tree contains MGTEvent objects
        event = chain.event
        
        if wfmCounter>numPulses:
            break
        
        #print "Elapsed time is %f" % ((event.GetTime() - startTime)/CLHEP.s)
    
        #Loop over each waveform in the event
        for i_wfm in xrange( event.GetNWaveforms()): #event.GetNWaveforms() ):
            digiData = event.GetDigitizerData(i_wfm)
            
            
            if (digiData.GetChannel() != boardChannel): continue 


            #Get the waveform
            wf = event.GetWaveform(i_wfm)
            
            #set the last sample to the second to last sample (fixes weird max ADC issue)
#            wfLength = wf.GetLength()
#            wf.SetValue(wfLength, wf.At(wfLength-3))
#            wf.SetValue(wfLength-1, wf.At(wfLength-3))
#            wf.SetValue(wfLength-2, wf.At(wfLength-3))

            #Subtract the baseline from the waveform, in case this
            #hasn't been done already.
            baseline.TransformInPlace(wf)
            
            
            if not isTemplateSet:
                print "Setting the template"
                sp.MakeSimilarTo(wf);
                sp.SetTemplate(wf);
                isTemplateSet = 1
                #do i need to re-do it on the second pass?
            else:
                fitter.SetTemplate(sp);
                fitter.TransformInPlace(wf);
            sp.AddPulse(wf);


            wfmCounter+=1


    exFinder = MGWFExtremumFinder();
    exFinder.SetFindMaximum();
    exFinder.TransformInPlace( sp );
    sp /= exFinder.GetTheExtremumValue();


#    wfSum = wfSumList[iRange]
#    wfSum.SetData(superP.GetVectorData())
#    wfSum.MakeSimilarTo(superP)
#
#    #wfSpr = MGTWaveform()
#    #wfSpr.SetData(superP.GetVariance().GetVectorData())
#
#    timePointCalc.FindTimePoints(superP);
#    tenTime = timePointCalc.GetFromStartRiseTime(0);
#    ninetyTime = timePointCalc.GetFromStartRiseTime(1);
#    riseTime = ninetyTime - tenTime
#    riseTimeList.append(riseTime)
#    maxAmpList.append(timePointCalc.GetTheExtremumValue())
#    print("\n>>>done with Range %d with risetime %f ns" % (iRange, riseTime))


    canvas = TCanvas("canvas", "A/E Peak Superpulses ",700,450)

    pad2 = TPad("pad2","The Superpulse", 0.05,0.05,0.95,0.95)
    gStyle.SetOptStat(0)
    pad2.Draw()
    pad2.cd()
#    wfSumList[0] /= float(maxAmpList[0])
#    
#    
#    print "WF sampling period is %f " % wfSumList[0].GetSamplingPeriod()

    ftHist = sp.GimmeHist().Clone()
    ftHist.SetTitle("Pulser Superpulse")
    ftHist.SetYTitle("AU")
    ftHist.Draw()

    tp = TPaveText(.55,.15,.85,.35, "blNDC")
    tp.SetTextSize(0.03);
    tp.SetTextAlign(12);
    tp.AddText("Averaged %d wfs" % sp.GetNPulses())
    #tp.AddText("  10-90 Risetime is %.2f ns" % riseTimeList[0])
    #tp.AddText("Post-Shift: Averaged %d wfs" %  superPList[1].GetNPulses())
    #tp.AddText("  10-90 Risetime is %.2f ns" % riseTimeList[1])
    #percentDiff = math.fabs(riseTimeList[1] - riseTimeList[0])/((riseTimeList[1] + riseTimeList[0])/2) * 100
    #tp.AddText("Percent Diff is %.2f" % percentDiff)
    tp.SetBorderSize(1);
    tp.Draw();
    canvas.Update()


#    # get vector data
#    wfData = wfSumList[0].GetVectorData()[0:30]
#    print "wf start length is " + str(len(wfData))
#    
#    waveform_vector = numpy.concatenate((wfData, wfSumList[0].GetVectorData()), axis=0)
#    print "wf length is " + str(len(waveform_vector))

    canvas.Update()

    leg = TLegend(0.125,0.7,0.42,0.85)
    leg.SetTextFont(72);
    leg.AddEntry(ftHist,"Raw Superpulse","l")
    leg.Draw();

    canvas.Update()

    canvas.Print(outFileName + "_plot.pdf" )
    canvas.Print(outFileName + "_plot.root")

    outfile = TFile(outFileName+".root", "RECREATE");
    outTree = TTree("pulser_superpulse", "Averaged Pulser Events")
    outTree.Branch("PulserWaveform", "MGTWaveform", sp);
    
    outTree.Fill()
    outfile.Write()
    outfile.Close()

#    value = raw_input('  --> Press return to quit : ')

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
