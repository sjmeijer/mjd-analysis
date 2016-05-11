#!/usr/bin/python
import ROOT
from ROOT import *
TROOT.gApplication.ExecuteFile("$MGDODIR/Root/LoadMGDOClasses.C")
TROOT.gApplication.ExecuteFile("$MGDODIR/Majorana/LoadMGDOMJClasses.C")
TROOT.gApplication.ExecuteFile("$GATDIR/LoadGATClasses.C")

import numpy as np
import matplotlib.pyplot as plt
import sys,os

from pymc3 import *
import theano.tensor as T
from theano.compile.ops import as_op


from scipy import ndimage, signal


def main(argv):

  plt.ion()
  wfFig = plt.figure(figsize=(15,10))


  #external pulser
#  channel = 674
#  runNumber = 6964
#  pulserEnergy = 1797 #defines center of energy cut
#  pulserRes = 1 #defines +/- around pulser energy to cut

  #internal pulser
  runNumber = 6890
#  channel = 674
#  #  pulserEnergy = 640 #defines center of energy cut
#  pulserRes = 10 #defines +/- around pulser energy to cut
  channel = 624
#  pulserEnergy = 600 #defines center of energy cut
#  pulserRes = 100 #defines +/- around pulser energy to cut

  # Define your cuts
  #energyCut = "trapECal>%f && trapECal<%f" % (pulserEnergy-pulserRes, pulserEnergy+pulserRes);
  channelCut = "channel == " + str(channel)
  energyCut = "trapECal>%f" % (100);
  #aeCut = " abs(rcDeriv50nsMax/energy) > .00054 && abs(rcDeriv50nsMax/energy) < .00062  "
  cut = energyCut + " && "+ channelCut #+ " && " + aeCut
  print "The cuts will be: " + cut


  #Prepare the MGDO classes we will use

  #flat-time will be used for a baseline subtraction
  flatTime = 800;
  #Instantiate and prepare the baseline remover transform:
  baseline = MGWFBaselineRemover()
  baseline.SetBaselineTime(flatTime)
  
  
  gatFilePath =  os.path.expandvars("$MJDDATADIR/%s/data/gatified/%s/%s%d.root" % (dataSetName, detectorName, gatDataName, runNumber  ) )
  builtFilePath =  os.path.expandvars("$MJDDATADIR/%s/data/built/%s/%s%d.root" % (dataSetName, detectorName, builtDataName, runNumber  ) )
  
  gat_file = TFile(gatFilePath)
  gatTree = gat_file.Get(gatTreeName)
  
  built_file = TFile(builtFilePath)
  builtTree = built_file.Get(builtTreeName)
  
  gatTree.SetEntryList(0)
  gatTree.Draw(">>elist", cut, "entrylist")
  elist = gDirectory.Get("elist")
  print "Number of entries in the entryList is " + str(elist.GetN())
  gatTree.SetEntryList(elist);
  builtTree.SetEntryList(elist);
  
  for ientry in xrange( elist.GetN() ):
    entryNumber = gatTree.GetEntryNumber(ientry);
    waveform = getWaveform(gatTree, builtTree, entryNumber, channel)

    #there are goofy things that happen at the end of the waveform for mod 1 data because of the presumming.  Just kill the last 5 samples
    waveform.SetLength(waveform.GetLength()-5)
    
    #for now, baseline subtract it here (should eventually be incorporated into the model.  won't be hard.  just lazy.)
    baseline.TransformInPlace(waveform)



    fitWaveform(waveform, wfFig)

#    value = raw_input('  --> Press s to skip,  q to quit, any other key to continue\n')
#    if value == 'q':
#      exit(0)


def getWaveform(gatTree, builtTree, entryNumber, channelNumber):
    
    builtTree.GetEntry( entryNumber )
    gatTree.GetEntry( entryNumber )
    
    event = builtTree.event
    channelVec   = gatTree.channel
    numWaveforms = event.GetNWaveforms()

    for i_wfm in xrange( numWaveforms ):
        channel = channelVec[i_wfm]
        if (channel != channelNumber): continue
        return event.GetWaveform(i_wfm)

def CreateSignalModelRLC(data, t0_guess, energy_guess, rcdiff_guess):
  
  with Model() as signal_model:
  
    t0 = DiscreteUniform('switchpoint', lower = t0_guess-10, upper=t0_guess+10, testval=t0_guess)
    wfScale = Normal('wfScale', mu=energy_guess, sd=.01*energy_guess)
    rc_diff = Normal('rc_diff', mu=rcdiff_guess, sd=100.) #also in ns
    
    rc = Wald('preampRC', mu=5.)
    lc = Wald('preampLC', mu=5.)
    
 
    @as_op(itypes=[T.lscalar, T.dscalar, T.dscalar, T.dscalar, T.dscalar], otypes=[T.dvector])
    def pulser_model(s, e, rc, lc, fall_time):

      step_wf = np.zeros_like(data)
      step_wf[s:] = 1.
      
      step_rlc = RLCFilter(step_wf, rc, lc)
      
      step_wf_decayed = RcDifferentiate(step_rlc, fall_time)
      
      model_wf = step_wf_decayed*e
      
      return model_wf

#    baseline_model = siggen_model(t0, radEst, phiEst, zEst, wfScale, tempEst, rc_int, rc_diff, rc_diff_short, fall_time_short_frac, detectorListIdx)
    baseline_model = pulser_model(t0,  wfScale, rc, lc,  rc_diff)
    baseline_observed = Normal("baseline_observed", mu=baseline_model, sd=10., observed= data )
  return signal_model

def CreateSignalModelOpAmp(data, t0_guess, energy_guess, rcdiff_guess):
  
  with Model() as signal_model:
  
    t0 = DiscreteUniform('switchpoint', lower = t0_guess-10, upper=t0_guess+10, testval=t0_guess)
    wfScale = Normal('wfScale', mu=energy_guess, sd=.01*energy_guess)
    rc_diff = Normal('rc_diff', mu=rcdiff_guess, sd=100.) #also in ns
    
    r1 = Wald('preampR1', mu=5.)
    r2 = Wald('preampR2', mu=5.)
    c = Wald('preampC', mu=5.)
    
 
    @as_op(itypes=[T.lscalar, T.dscalar, T.dscalar, T.dscalar,T.dscalar, T.dscalar], otypes=[T.dvector])
    def pulser_model(s, e, r1, r2, c, fall_time):

      step_wf = np.zeros_like(data)
      step_wf[s:] = 1.
      
      step_rlc = OpAmpFilter(step_wf, r1, r2, c)
      
      step_wf_decayed = RcDifferentiate(step_rlc, fall_time)
      
      model_wf = step_wf_decayed*e
      
      return model_wf

#    baseline_model = siggen_model(t0, radEst, phiEst, zEst, wfScale, tempEst, rc_int, rc_diff, rc_diff_short, fall_time_short_frac, detectorListIdx)
    baseline_model = pulser_model(t0,  wfScale, r1, r2, c,  rc_diff)
    baseline_observed = Normal("baseline_observed", mu=baseline_model, sd=10., observed= data )
  return signal_model

def CreateSignalModelFoldedCascode(data, t0_guess, energy_guess, rcdiff_guess):
  
  with Model() as signal_model:
  
    t0 = DiscreteUniform('switchpoint', lower = t0_guess-10, upper=t0_guess+10, testval=t0_guess)
    wfScale = Normal('wfScale', mu=energy_guess, sd=.01*energy_guess)
    rc_diff = Normal('rc_diff', mu=rcdiff_guess, sd=100.) #also in ns
    
    c1 = Wald('preampC1', mu=5.)
    c2 = Wald('preampC2', mu=5.)
    r = Wald('preampR', mu=5.)
    
 
    @as_op(itypes=[T.lscalar, T.dscalar, T.dscalar, T.dscalar,T.dscalar, T.dscalar], otypes=[T.dvector])
    def pulser_model(s, e, c1, c2, r, fall_time):

      step_wf = np.zeros_like(data)
      step_wf[s:] = 1.
      
      step_rlc = FoldedCascodeFilter(step_wf, c1, c2, r)
      
      step_wf_decayed = RcDifferentiate(step_rlc, fall_time)
      
      model_wf = step_wf_decayed*e
      
      return model_wf

#    baseline_model = siggen_model(t0, radEst, phiEst, zEst, wfScale, tempEst, rc_int, rc_diff, rc_diff_short, fall_time_short_frac, detectorListIdx)
    baseline_model = pulser_model(t0,  wfScale, c1, c2, r,  rc_diff)
    baseline_observed = Normal("baseline_observed", mu=baseline_model, sd=10., observed= data )
  return signal_model




def fitWaveform(wf, wfFig):

  np_data = wf.GetVectorData()
  wfMax = np.amax(np_data)

  lastFitSampleIdx = np.argmax(np_data)+50

  startGuess = findTimePoint(np_data, 0.05)
  firstFitSampleIdx = startGuess-25
  fitSamples = lastFitSampleIdx-firstFitSampleIdx # 1 microsecond
  t0_guess = startGuess - firstFitSampleIdx
  
  np_data_early = np_data[firstFitSampleIdx:lastFitSampleIdx]

#  pulser_model = CreateSignalModelRLC(np_data_early, t0_guess, wfMax, 72 * CLHEP.us)

#  pulser_model = CreateSignalModelOpAmp(np_data_early, t0_guess, wfMax, 72 * CLHEP.us)

  pulser_model = CreateSignalModelFoldedCascode(np_data_early, t0_guess, wfMax, 72 * CLHEP.us)

  plt.figure(wfFig.number)
  plt.clf()
  plt.title("Charge waveform")
  plt.xlabel("Sample number [10s of ns]")
  plt.ylabel("Raw ADC Value [Arb]")
  plt.plot( np_data  ,color="red" )
  plt.xlim( lastFitSampleIdx - fitSamples, lastFitSampleIdx+20)
  plt.show()
  
  value = raw_input('  --> Press s to skip,  q to quit, any other key to continue with fit\n')
  if value == 'q':
      exit(0)
  if value == 's':
      return
  
  with pulser_model:

    step = Slice()
    
    trace = sample(1000, step)
    burnin = 800

    t0 = np.around( np.median(  trace['switchpoint'][burnin:]))
    scale =         np.median(  trace['wfScale'][burnin:])
    fall_time =     np.median(  trace['rc_diff'][burnin:])
    
    c1 =         np.median(  trace['preampC1'][burnin:])
    c2 =         np.median(  trace['preampC2'][burnin:])
    r =         np.median(  trace['preampR'][burnin:])

    
#    r1 =         np.median(  trace['preampR1'][burnin:])
#    r2 =         np.median(  trace['preampR2'][burnin:])
#    c =         np.median(  trace['preampC'][burnin:])
#    rc =         np.median(  trace['preampRC'][burnin:])
#    lc =     np.median(  trace['preampLC'][burnin:])

    
    startVal = t0 + firstFitSampleIdx
    
    print "<<<startVal is %d" % startVal
    print "<<<scale is %0.2f" % scale
    print "<<<rc_diff is %0.2f" % (fall_time/1000.)
    
    print "<<<c1 is %0.2f" % c1
    print "<<<c2 is %0.2f" % c2
    print "<<<r is %0.2f" % r


    plt.ioff()
    traceplot(trace)
    plt.ion()
  
  
  step_wf = np.zeros_like(np_data_early)
  step_wf[t0:] = 1.
#  step_1 = RLCFilter(step_wf, rc, lc)
#  step_1 = OpAmpFilter(step_wf, r1, r2, c)
  step_1 = FoldedCascodeFilter(step_wf, c1, c2, r)

  step_wf_decayed = RcDifferentiate(step_1, fall_time)
  model_wf = step_wf_decayed*scale


  plt.figure(wfFig.number)
  plt.plot(np.arange(firstFitSampleIdx, len(model_wf)+firstFitSampleIdx), model_wf  ,color="blue" )
  plt.show()
  
  value = raw_input('  --> Press q to quit, any other key to continue\n')
  if value == 'q':
    exit(1)


def RLCFilter(anInput, rc, lc):
  num = [1]
  den = [lc, rc, 1]
  system = signal.TransferFunction(num, den)
  t = np.arange(0, len(anInput))
  
  tout, y, x = signal.lsim(system, anInput, t)

  return y

  
def OpAmpFilter(anInput, r1, r2, c):
  num = [r2]
  den = [r1*r2*c, r1]
  system = signal.TransferFunction(num, den)
  t = np.arange(0, len(anInput))
  
  tout, y, x = signal.lsim(system, anInput, t)

  return y

def FoldedCascodeFilter(anInput, c1, c2, r):
  num = [c1*r,0]
  den = [c2*r, 1]
  system = signal.TransferFunction(num, den)
  t = np.arange(0, len(anInput))
  
  tout, y, x = signal.lsim(system, anInput, t)

  return y

def RcDifferentiate(anInput, timeConstantInNs):
    timeConstantInSamples = timeConstantInNs / 10.
    dummy = anInput[0];
    anOutput = np.copy(anInput)
    dummy2 = 0.0;
    for i in xrange(1,len(anInput)):
      dummy2  = anOutput[i-1] + anInput[i] - dummy - anOutput[i-1] / timeConstantInSamples;
      dummy = anInput[i];
      anOutput[i] = dummy2;
   
   
    anOutput /= np.amax(anOutput)
    return anOutput

def findTimePoint(data, percent):
  #don't screw up the data, bro
  int_data = np.copy(data)
  int_data /= np.amax(int_data)
  return np.where(np.greater(int_data, percent))[0][0]

gatDataName = "mjd_run"
gatTreeName = "mjdTree"
builtDataName = "OR_run"
builtTreeName = "MGTree"
dataSetName = "surfmjd"
detectorName = "P3JDY"

if __name__=="__main__":
    main(sys.argv[1:])

