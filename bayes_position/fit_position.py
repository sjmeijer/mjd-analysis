#!/usr/local/bin/python
from ROOT import *
TROOT.gApplication.ExecuteFile("$MGDODIR/Root/LoadMGDOClasses.C")
TROOT.gApplication.ExecuteFile("$MGDODIR/Majorana/LoadMGDOMJClasses.C")
import matplotlib
#matplotlib.use('CocoaAgg')
import sys, os
import matplotlib.pyplot as plt

import numpy as np

#import pymc
#import signal_model as sm
from pymc3 import *
import signal_model_pymc3 as sm3
from detector_model import *

#plt.style.use('presentation')

gatDataName = "mjd_run"
gatTreeName = "mjdTree"
builtDataName = "OR_run"
builtTreeName = "MGTree"
dataSetName = "surfmjd"
detectorName = "P3JDY"

#det = Detector("P42664A_0p05grad.conf", 41.5, 38.4, 51.6* CLHEP.ns, 71.8*CLHEP.us, 1.65*CLHEP.us,0.009, zeroPadding=500, gaussian_smoothing=0)
#channelNumber = 674

channelNumber = 624

#gradList = np.arange(5, 10, 1)
#pcRadList = np.arange(2.25, 3.05, 0.1)
#gradientRange = np.arange(0.05, 0.07, 0.0025)
#pcRadiusRange = np.arange(2.55, 2.75, 0.025)

#gradList = np.arange(0.0500, 0.0700, 0.0025)
#pcRadList = np.arange(2.350, 2.650, 0.025)

gradList = np.arange(0.01, 0.09, 0.01)
pcRadList = np.arange(1.65, 2.95, 0.1)


detArray  = np.empty( (len(gradList),len(pcRadList)), dtype=object)

for (gradIdx,grad) in enumerate(gradList):
#  grad /= 10000.
  for (radIdx, pcRad) in enumerate(pcRadList):
#    detName = "P42574A_grad%0.3f_pcrad%0.4f.conf" % (grad,pcRad)
    detName = "P42574A_grad%0.2f_pcrad%0.2f.conf" % (grad,pcRad)
    det =  Detector(detName, 39.3, 33.8, 66.5* CLHEP.ns, 71.8*CLHEP.us, zeroPadding=500, temperature=99.)
    detArray[gradIdx, radIdx] = det

#  lookupName = "P42574A_grad%0.2f_lookup.py" % grad
#  det.LoadLookupTable(lookupName)



#det.LoadLookupTable("P42664A_normal_fine.npy")
flatTimeSamples = 800


####################################################################################################################################################################


def main(argv):
  runRange = (6006,6360)

  plt.ion()
  fig = plt.figure(1)
  fig2=None
#  fig2 = plt.figure(2)

  #Instantiate and prepare the baseline remover transform:
  baseline = MGWFBaselineRemover()
  baseline.SetBaselineSamples(flatTimeSamples)


  #i do this one run at a time, instead of in a chain, because it makes it easier when we want to run on large data sets and create skim files for each run
  for iRun in range( runRange[0],  runRange[1]+1):
    print 'processing run', iRun
    gatFilePath =  os.path.expandvars("$MJDDATADIR/%s/data/gatified/%s/%s%d.root" % (dataSetName, detectorName, gatDataName, iRun  ) )
    builtFilePath =  os.path.expandvars("$MJDDATADIR/%s/data/built/%s/%s%d.root" % (dataSetName, detectorName, builtDataName, iRun  ) )
    
    if not os.path.isfile(gatFilePath):
      print ">>>Skipping file " + gatFilePath
      continue
  
    gat_file = TFile.Open(gatFilePath)
    gatTree = gat_file.Get(gatTreeName)
    built_file = TFile.Open(builtFilePath)
    builtTree = built_file.Get(builtTreeName)
    
    builtTree.AddFriend(gatTree)
    
    regAECut = 0.01425
    highAeCut = 0.0177
    
    chanCut =  "channel == %d" % channelNumber
    energyCut = "trapECal>%f && trapECal<%f" % (1588,1594)
    #energyCut = "trapECal>%f" % 1500
    aeCut = "TSCurrent100nsMax/trapECal > %f" % regAECut

    cut = energyCut + " && " + chanCut + " && " + aeCut
    
    
    #print "The cuts will be: " + cut

    gatTree.SetEntryList(0)
    gatTree.Draw(">>elist", cut, "entrylist")
    elist = gDirectory.Get("elist")
    #print "Number of entries in the entryList is " + str(elist.GetN())
    gatTree.SetEntryList(elist);
    builtTree.SetEntryList(elist);
    
    for ientry in xrange( elist.GetN() ):
      entryNumber = gatTree.GetEntryNumber(ientry);
      waveform = getWaveform(gatTree, builtTree, entryNumber, channelNumber)

      #there are goofy things that happen at the end of the waveform for mod 1 data because of the presumming.  Just kill the last 5 samples
      waveform.SetLength(waveform.GetLength()-5)
      
      #for now, baseline subtract it here (should eventually be incorporated into the model.  won't be hard.  just lazy.)
      baseline.TransformInPlace(waveform)

#      det.preampRiseTime = 54
#      det.preampFallTime = 71.8 * 1000
      #MCMC fit and plot the results

      print "Waveform number %d in run %d" % (entryNumber, iRun)
      fitWaveform(waveform, fig, fig2)


####################################################################################################################################################################

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

####################################################################################################################################################################



def plotWaveform(wfFig, np_data_early, wfScale, offset):

  plt.figure(wfFig.number)
  plt.clf()
  plt.title("Charge waveform")
  plt.xlabel("Sample number [10s of ns]")
  plt.ylabel("Raw ADC Value [Arb]")
  
  plt.plot( np_data_early  ,color="red" )
  
  
  det = detArray[len(gradList)/2, len(pcRadList)/2]
  
  siggen_fit= det.GetSiggenWaveform(10, 0, 10, energy=1)
  siggen_fit = det.ProcessWaveform(siggen_fit)
  
  siggen_fit = siggen_fit[500::] #???
  
  siggen_fit *= wfScale

  plt.plot(np.arange(offset, len(siggen_fit)+offset), siggen_fit  ,color="blue" )
#  plt.axvline(x=lastFitSampleIdx, linewidth=1, color='r',linestyle=":")
#  plt.axvline(x=startGuess, linewidth=1, color='g',linestyle=":")

  plt.xlim(0, len(np_data_early))
  
  value = raw_input('  --> Press s to skip,  q to quit, any other key to continue with fit\n')
  if value == 'q':
    exit(0)
  if value == 's':
    return 0
  return 1

#def fitWaveformTester(wf, wfFig, zoomFig):
#  np_data = wf.GetVectorData()
#  
#  wfMax = np.amax(np_data)
#  lastFitSampleIdx = np.argmax(np_data)+30#1100#findTimePoint(np_data, 0.99)
#  
#  startGuess = findTimePoint(np_data, 0.005)
#  
#  firstFitSampleIdx = startGuess-30
#  
#  fitSamples = lastFitSampleIdx-firstFitSampleIdx # 1 microsecond
#  
#  #startGuess = 973
#  t0_guess = startGuess - firstFitSampleIdx
#  
#  
#  np_data_early = np_data[firstFitSampleIdx:lastFitSampleIdx]
#
#  siggen_model = sm3.CreateCheapDetectorModel(det, np_data_early, t0_guess, wfMax)
#
#  with siggen_model:
##    start = find_MAP(fmin=optimize.fmin_powell)
#    trace = sample(20000)
#
#    traceplot(trace);
#    
#    burnin = 15000
#    t0 = np.around( np.median(  trace['switchpoint'][burnin:]))
#    scale =         np.median(  trace['wfScale'][burnin:])
#    
#    siggen_fit= np.zeros_like(np_data_early)
#    
#    
#    f = plt.figure()
#    plt.plot(np_data_early, color="red")
#    plt.plot(siggen_fit, color="blue")
#
#    plt.show()

def fitWaveform(wf, wfFig, zoomFig):

  np_data = wf.GetVectorData()
  wfMax = np.amax(np_data)

  #perform the fit up to this index.  Currently set by 99% timepoint (no real point in fitting on the falling edge)
  lastFitSampleIdx = np.argmax(np_data)+25

#  lastFitSampleIdx = findTimePoint(np_data, .95)
  startGuess = findTimePoint(np_data, 0.005)
  firstFitSampleIdx = startGuess-20
  fitSamples = lastFitSampleIdx-firstFitSampleIdx # 1 microsecond
  t0_guess = startGuess - firstFitSampleIdx
  
  np_data_early = np_data[firstFitSampleIdx:lastFitSampleIdx]

  plt.ion()

  doFit = plotWaveform(wfFig, np_data_early, wfMax, t0_guess)

  if not doFit:
    return 0
  
  det = detArray[len(gradList)/2, len(pcRadList)/2]
  siggen_model = sm3.CreateFullDetectorModel(detArray, np_data_early, t0_guess, wfMax, det.preampRiseTime, det.preampFalltimeLong)
  
  with siggen_model:
#    start = find_MAP(fmin=optimize.fmin_powell)
    #step = Slice()
    step = Metropolis()
    
    #for M-H
#    one_minute = 0.25 * 1E3 #500 takes about 2 minutes
#    one_hour = 60 * one_minute
#    trace = sample(30*one_minute, step)
#    burnin = 20*one_minute

    # for slice
    one_minute = np.around(5820 / 21.3)
    one_hour = 60 * one_minute
    trace = sample(10*one_hour, step)
    burnin = 9*one_hour


    t0 = np.around( np.median(  trace['switchpoint'][burnin:]))
    r =             np.median(  trace['radEst'][burnin:])
    z =             np.median(  trace['zEst'][burnin:])
    phi =           np.median(  trace['phiEst'][burnin:])
    scale =         np.median(  trace['wfScale'][burnin:])
    temp =         np.median(  trace['temp'][burnin:])
    rise_time =     np.median(  trace['rc_int'][burnin:])
    fall_time =     np.median(  trace['rc_diff'][burnin:])
    gaussSmooth =     np.median(  trace['gaussSmooth'][burnin:])
#    fall_time_short =     np.median(  trace['rc_diff_short'][burnin:])
#    fall_time_frac =     np.median(  trace['rc_diff_short_frac'][burnin:])

    gradIdx =        np.median( trace['gradIdx'][burnin:])
    pcRadIdx =       np.median( trace['pcRadIdx'][burnin:])
    
    
    startVal = t0 + firstFitSampleIdx
    
    print "<<<startVal is %d" % startVal
    print "<<<r is %0.2f" % r
    print "<<<z is %0.2f" % z
    print "<<<phi is %0.2f" % phi
    print "<<<scale is %0.2f" % scale
    print "<<<detector gradient is %0.2f (idx is %d)" % (gradList[gradIdx], gradIdx)
    print "<<<PC Radius is %0.2f (idx is %d)" % (pcRadList[pcRadIdx], pcRadIdx)
    print "<<<Gauss smooth is %0.2f " % ( gaussSmooth )

    print "<<<rc_int is %0.2f" % rise_time
    print "<<<rc_diff is %0.2f" % (fall_time/1000.)
    print "<<<temp is %0.2f" % (temp)

    plt.ioff()
    traceplot(trace)
    plt.ion()
  
#    # Two subplots, the axes array is 1-d
#    f, axarr = plt.subplots(7, sharex=True)
#    
#    axarr[0].plot(trace['radEst'][:])
#    axarr[0].set_ylabel('r [mm]')
#    axarr[1].plot(trace['zEst'][:])
#    axarr[1].set_ylabel('z [mm]')
#    axarr[2].plot(trace['phiEst'][:])
#    axarr[2].set_ylabel('phi [rad]')
#    axarr[3].plot(trace['wfScale'][:])
#    axarr[3].set_ylabel('ADCscale [Arb.]')
#    axarr[4].plot(trace['switchpoint'][:])
#    axarr[4].set_ylabel('t_0 [s]')
#    axarr[5].plot(trace['rc_int'][:])
#    axarr[5].set_ylabel('rc int [ns]')
#    axarr[6].plot(trace['rc_diff'][:])
#    axarr[6].set_ylabel('rc diff [ns]')
#    axarr[0].set_xlabel('MCMC Step Number')
#    axarr[0].set_title('Raw MCMC Sampling')
#  

#  
#  siggen_model = pymc.Model( sm.CreateFullDetectorModel(det, np_data_early, t0_guess, wfMax, det.preampRiseTime, det.preampFallTime) )
# 
##  M = pymc.MCMC(siggen_model)
##  M.use_step_method(pymc.AdaptiveMetropolis, [M.radEst, M.zEst, M.phiEst, M.wfScale, M.switchpoint], delay=50000)
###  M.use_step_method(pymc.AdaptiveMetropolis, [M.radEst, M.zEst, M.phiEst, M.wfScale], delay=1000)
###  M.use_step_method(pymc.DiscreteMetropolis, M.switchpoint, proposal_distribution='Normal', proposal_sd=4)
#
#  M = pymc.MCMC(siggen_model)
#  M.use_step_method(pymc.DiscreteMetropolis, M.switchpoint, proposal_sd=.5, proposal_distribution='Normal')
#  M.use_step_method(pymc.Metropolis, M.wfScale, proposal_sd=5., proposal_distribution='Normal')
#  M.use_step_method(pymc.AdaptiveMetropolis, [M.radEst, M.zEst, M.phiEst, M.wfScale, M.rc_int, M.rc_diff], shrink_if_necessary=True)
##  M.use_step_method(pymc.Metropolis, M.radEst, proposal_sd=.5, proposal_distribution='Normal')
##  M.use_step_method(pymc.Metropolis, M.zEst, proposal_sd=.5, proposal_distribution='Normal')
##  M.use_step_method(pymc.Metropolis, M.phiEst, proposal_sd=np.pi/16, proposal_distribution='Normal')
#
#
#  M.sample(iter=10000)
#  M.db.close()
# 
#  burnin = 4000
# 
#  t0 = np.around( np.median(M.trace('switchpoint')[burnin:]))
#  r =  np.median(M.trace('radEst')[burnin:])
#  z =  np.median(M.trace('zEst')[burnin:])
#  phi =  np.median(M.trace('phiEst')[burnin:])
#  scale =  np.median(M.trace('wfScale')[burnin:])
#  rise_time =  np.median(M.trace('rc_int')[burnin:])
#  fall_time =  np.median(M.trace('rc_diff')[burnin:])
#  startVal = t0 + firstFitSampleIdx
#  
#  print "<<<startVal is %d" % startVal
#  print "<<<r is %0.2f" % r
#  print "<<<z is %0.2f" % z
#  print "<<<phi is %0.2f" % phi
#  print "<<<scale is %0.2f" % scale
#  print "<<<rc_int is %0.2f" % rise_time
#  print "<<<rc_int is %0.2f" % (fall_time/1000.)
#  
#  # Two subplots, the axes array is 1-d
#  f, axarr = plt.subplots(7, sharex=True)
#  
#  axarr[0].plot(M.trace('radEst')[:])
#  axarr[0].set_ylabel('r [mm]')
#  axarr[1].plot(M.trace('zEst')[:])
#  axarr[1].set_ylabel('z [mm]')
#  axarr[2].plot(M.trace('phiEst')[:])
#  axarr[2].set_ylabel('phi [rad]')
#  axarr[3].plot(M.trace('wfScale')[:])
#  axarr[3].set_ylabel('ADCscale [Arb.]')
#  axarr[4].plot(M.trace('switchpoint')[:])
#  axarr[4].set_ylabel('t_0 [s]')
#  axarr[5].plot(M.trace('rc_int')[:])
#  axarr[5].set_ylabel('rc int [ns]')
#  axarr[6].plot(M.trace('rc_diff')[:])
#  axarr[6].set_ylabel('rc diff [ns]')
#  axarr[0].set_xlabel('MCMC Step Number')
#  axarr[0].set_title('Raw MCMC Sampling')


#  t0 = t0_guess
#  r =  30
#  z =  13
#  phi =  .74
#  scale =  3990
#  startVal = startGuess

  
  print ">>> t0guess was %d" % (firstFitSampleIdx+t0_guess)
  print ">>> fit t0 was %d" % (firstFitSampleIdx + t0)

  det = detArray[np.int(gradIdx), np.int(pcRadIdx)]
  
  det.preampRiseTime = rise_time
  det.preampFallTime = fall_time
  det.gaussian_smoothing = gaussSmooth
#  det.preampFalltimeShort = fall_time_short
#  det.preampFalltimeShortFraction = fall_time_frac
  det.SetTemperature(temp)
  siggen_fit= det.GetSiggenWaveform(r, phi, z, energy=2600)
  siggen_fit = det.ProcessWaveform(siggen_fit)
  siggen_fit = siggen_fit[500::]
  
  siggen_fit *= scale
  
  
  np.save("fit_waveform.npy", siggen_fit)
  
  #plotting

  plt.figure(wfFig.number)
  plt.clf()
  plt.title("Charge waveform")
  plt.xlabel("Sample number [10s of ns]")
  plt.ylabel("Raw ADC Value [Arb]")
  plt.plot( np_data  ,color="red" )
  plt.xlim( lastFitSampleIdx - fitSamples, lastFitSampleIdx+20)

#  plt.plot(np.arange(startVal, len(siggen_fit)+startVal), siggen_fit  ,color="blue" )
  for idx in np.arange(burnin, len(trace['rc_diff']), 1):
#    print "plotting number %d" % idx
    t0 = np.around( trace['switchpoint'][idx])
    r =             trace['radEst'][idx]
    z =             trace['zEst'][idx]
    phi =           trace['phiEst'][idx]
    scale =         trace['wfScale'][idx]
    temp =          trace['temp'][idx]
    rise_time =     trace['rc_int'][idx]
    fall_time =     trace['rc_diff'][idx]
    gaussSmooth =   trace['pcRadIdx'][idx]
    gradIdx =       trace['gradIdx'][idx]
    gaussSmooth =    trace['gaussSmooth'][idx]
    
    startVal = t0 + firstFitSampleIdx
    det = detArray[np.int(gradIdx), np.int(pcRadIdx)]
    det.preampRiseTime = rise_time
    det.preampFallTime = fall_time
    det.gaussian_smoothing = gaussSmooth
    det.SetTemperature(temp)
    siggen_fit_idx = det.GetSiggenWaveform(r, phi, z, energy=2600)
    siggen_fit_idx = det.ProcessWaveform(siggen_fit_idx)
    siggen_fit_idx = siggen_fit_idx[500::]
    
    siggen_fit_idx *= scale

    plt.plot(np.arange(startVal, len(siggen_fit_idx)+startVal), siggen_fit_idx  ,color="blue", alpha = 0.1)
#    plt.show()
#    value = raw_input('  --> Press q to quit, any other key to continue\n')
#
#    if value == 'q':
#      exit(1)

#  plt.xlim( startVal-10, startVal+10)
#  plt.ylim(-10, 25)
  plt.plot(np.arange(startVal, len(siggen_fit)+startVal), siggen_fit  ,color="purple" )
#  plt.plot(np.arange(0, startVal), np.zeros(startVal)  ,color="blue" )
  plt.axvline(x=lastFitSampleIdx, linewidth=1, color='r',linestyle=":")
  plt.axvline(x=startVal, linewidth=1, color='g',linestyle=":")
  
#  rad_arr = M.trace('radEst')[burnin:]
#  z_arr = M.trace('zEst')[burnin:]
#  phi_arr = M.trace('phiEst')[burnin:]
#  n_bins=25

  
#  fig0 = plt.figure(6)
#  weights = np.ones_like(rad_arr)/float(len(rad_arr))
#  n, bins, patches = plt.hist(rad_arr, n_bins, histtype='step', linewidth=5, weights=weights)
#  plt.xlabel("r value [mm]")
#  plt.ylabel("probability")
#  plt.savefig("pdf_r.pdf")
#  
#  fig0 = plt.figure(7)
#  weights = np.ones_like(z_arr)/float(len(z_arr))
#  n, bins, patches = plt.hist(z_arr, n_bins, histtype='step', linewidth=5, weights=weights)
#  plt.xlabel("z value [mm]")
#  plt.ylabel("probability")
#  plt.savefig("pdf_z.pdf")
#  
#  fig0 = plt.figure(8)
#  weights = np.ones_like(phi_arr)/float(len(phi_arr))
#  n, bins, patches = plt.hist(phi_arr, n_bins, histtype='step', linewidth=5, weights=weights)
#  plt.xlabel("phi value [rad]")
#  plt.ylabel("probability")
#  plt.savefig("pdf_phi.pdf")

  
#  plt.figure(zoomFig.number)
#  plt.clf()
#  plt.title("Zoom in near the start time")
#  plt.plot( np_data  ,color="red" )
#  plt.xlim( startVal-40, startVal+40)
#  plt.ylim(-10, 50)
#  plt.plot(np.arange(startVal, 800+startVal), sigFit  ,color="blue" )
#  plt.plot(np.arange(0, startVal), np.zeros(startVal)  ,color="blue" )
#  plt.axvline(x=startVal, linewidth=1, color='g',linestyle=":")


  plt.show()
  value = raw_input('  --> Press q to quit, any other key to continue\n')

  if value == 'q':
    exit(1)



def findTimePoint(data, percent):
  #don't screw up the data, bro
  int_data = np.copy(data)
  int_data /= np.amax(int_data)
  return np.where(np.greater(int_data, percent))[0][0]

#def findSiggenWaveform(x,y,z,normMax):
#
#  weightedClusterPosition = TVector3(x, y, z);
#  clusterWaveform = MGTWaveform();
#  calcFlag = siggenInst.CalculateWaveform(weightedClusterPosition, clusterWaveform, 1);
#  
#  rcint = MGWFRCIntegration()
#  rcint.SetTimeConstant(rcIntTimeConstant)
#  rcint.TransformInPlace(clusterWaveform)
#  
#  siggen_data = clusterWaveform.GetVectorData()
#  siggen_data *= normMax / np.amax(siggen_data)
#
#  return siggen_data





if __name__=="__main__":
    main(sys.argv[1:])


