#!/usr/local/bin/python
from ROOT import *
TROOT.gApplication.ExecuteFile("$MGDODIR/Root/LoadMGDOClasses.C")
TROOT.gApplication.ExecuteFile("$MGDODIR/Majorana/LoadMGDOMJClasses.C")
#ROOT.gApplication.ExecuteFile("$DISSDIR/Data/figure_style.C")

import matplotlib
matplotlib.use('pdf') #use a non-interactive backend that can handle multiprocessing
import matplotlib.pyplot as plt

import sys, os, array, pymc, time
import numpy as np
from scipy import ndimage
import slowpulse_model_mc2 as sm
import multiprocessing as mp

import wavelet_denoise_malbek as denoiser

#Graham's directory prefix
#dirPrefix = '$DISSDIR/Data'

#Ben's directory prefix
dirPrefix = '$MJDDATADIR/malbek/'

numCores = 8
sleepTime = 1

doPlots=1
writeFile = 1
waveletDenoise = 1


newTreeName = "spParamSkim_190_wavelet.root"
outputDir = "output190_wavelet"


flatTimeSamples = 2000 #in number of samples, not time

def main(argv):

  setupDirs(outputDir, 0, 15)

  #Instantiate and prepare the baseline remover transform:
  baseline = MGWFBaselineRemover()
  baseline.SetBaselineSamples(flatTimeSamples)
  
#  #Load the malbek t4 tree without any RT cuts
#  file = TFile(dirPrefix + 'test1_t4.root')
#  tree_nort = file.Get('fTree')
#
#  #Load the malbek t4 waveform tree
#  wf_file = TFile(dirPrefix + 'test1_t4_wf.root')
#  tree_wf = wf_file.Get('tree')

  
  #Load the malbek t4 tree without any RT cuts
  file = TFile(dirPrefix + 'ThesisAll_newCal.root')
  tree_nort = file.Get('fTree_noRT')
  
  #Load the malbek t4 waveform tree
  wf_file = TFile(dirPrefix + 't4_wf_all.root')
  tree_wf = wf_file.Get('tree')

#  #Load the malbek t4 tree without any RT cuts
#  file = TFile(dirPrefix + 'LinearCrateSpecial.root')
#  tree_nort = file.Get('fTree_noRT')
#
#  #Load the malbek t4 waveform tree
#  wf_file = TFile(dirPrefix + 'pb_t4_wf.root')
#  tree_wf = wf_file.Get('tree')

  #Build an index for wf tree for later access
  tree_wf.BuildIndex('run_id', 'event_id')
  
  tree_nort.AddFriend(tree_wf)

  #Loop through events within a given energy range
  energy_low = 0.6
  energy_high = 15.
  count = 0
  energyCut = "rfEnergy_keV>%f && rfEnergy_keV<%f" % (energy_low,energy_high)
  cut = energyCut

  tree_nort.SetEntryList(0)
  tree_nort.Draw(">>elist", cut, "entrylist")
  elist = gDirectory.Get("elist")
  tree_nort.SetEntryList(elist);
  tree_wf.SetEntryList(elist);
  numEntries = elist.GetN()
  print "Total number of entries (w/ energy cut): %d" % numEntries

#  results_queue   = mp.JoinableQueue()
  manager = mp.Manager()
  results_list = manager.list()
  argsList = []
  
  energyList = np.empty(numEntries)
  riseTimeList = np.empty(numEntries)
  wparList = np.empty(numEntries)
  
  #numEntries = 10
  print "Pulling root data for %d events" % (numEntries)
  for i in xrange( numEntries):
    entryNumber = tree_nort.GetEntryNumber(i);
    #    entryNumber = i

    tree_nort.GetEntry(entryNumber)
    tree_wf.GetEntry(entryNumber)
  
    current_run = tree_nort.rfRunID
    current_id = tree_nort.rfEventID   
    if tree_wf.GetEntryWithIndex(current_run, current_id) == -1:
      print 'waveform %s, %s not found' % (current_run, current_id)
      break
    waveform = tree_wf.waveform.Clone()
          
    #Baseline subtract it here (should eventually be incorporated into the model?)
    baseline.TransformInPlace(waveform)
    
    wfParams = (tree_nort.rfEnergy_keV, tree_nort.rfWpar, tree_nort.rfRiseTime) # (energy, wpar, risetime -- just passing anything I want into the parallel process)

    argsList.append( (waveform, wfParams, entryNumber, results_list) )


  runningProcesses = []

  #start a job for each core (must be separate processes.  yucky.)
  for i in range(numCores):
    p = mp.Process(target = fitWaveform, args=argsList.pop())
    p.start()
    runningProcesses.append(p)
  counter = 0
  #keep numCores jobs running simultaneously
  while len(argsList) > 0:
      for p in runningProcesses:
        if not p.is_alive():
          p.join()
          runningProcesses.remove(p)
          counter +=1
          print "finished job number %d of %d" % (counter, numEntries)
          new_p = mp.Process(target = fitWaveform, args=argsList.pop())
          new_p.start()
          runningProcesses.append(new_p)
          if len(argsList) == 0: break
      print "straight loopin"
      time.sleep(sleepTime)

  while len(runningProcesses)>0:
    for p in runningProcesses:
      if not p.is_alive():
        p.join()
        runningProcesses.remove(p)
        counter +=1
        print "finished job number %d of %d" % (counter, numEntries)
  
  #pull results out of the queue
#  results = []
#  for _ in range(numEntries):
#    # indicate done results processing
#    results.append(results_queue.get())
#    results_queue.task_done()
#
#  results_queue.join()
#  results_queue.close()

#  for (i, result) in enumerate(results_list):
#    print "Result %d" % i
#    print "-->spParam %f" % result[0]
#    print "-->energy  %f" % result[1][0]
#    print "-->wpar    %f" % result[1][1]
#    print "-->rt      %f" % result[1][2]

  if writeFile:
    oFile = TFile(newTreeName,"recreate");
    oFile.cd();
    outTree = TTree("slowpulseParamTree","Slowpulse Param");
    energy = np.zeros(1, dtype=float)
    spParam = np.zeros(1, dtype=float)
    spParam5 = np.zeros(1, dtype=float)
    spParam10 = np.zeros(1, dtype=float)
    riseTime = np.zeros(1, dtype=float)
    wpar = np.zeros(1, dtype=float)
    outTree.Branch("energykeV",energy, "energykeV/D");
    outTree.Branch("spParam",spParam, "spParam/D");
    outTree.Branch("spParam5",spParam5, "spParam/D");
    outTree.Branch("spParam10",spParam10, "spParam/D");
    outTree.Branch("riseTime",riseTime, "riseTime/D");
    outTree.Branch("wpar",wpar, "wpar/D");

    for (iEntry, result) in enumerate(results_list):
      energy[0] = result[1][0]
      spParam[0] = result[0][0]
      spParam5[0] = result[0][1]
      spParam10[0] = result[0][2]
      riseTime[0] = result[1][2]
      wpar[0] = result[1][1]
      outTree.Fill();
    oFile.Write();
    oFile.Close()


def fitWaveform( wf, wfParams, entryNumber, results_list ):
  
  f = open(os.devnull, 'w')
  sys.stdout = f
  
  np_data = wf.GetVectorData()
  
#  #TODO: find noise characteristics BEFORE denoising?
#  noise_sigma_guess = np.std(np_data[0:flatTimeSamples])

  if waveletDenoise:
    np_data_unfiltered = np_data
    np_data = denoiser.denoise_waveform(np_data, flatTimeSamples)

  energy = wfParams[0]

  #wfMax = np.amax(np_data)
  
  lastFitSampleIdx = 4300
  fitSamples = 2000 #can't be longer than 800 right now (that's the length of the siggen wf...)

  firstFitSampleIdx = lastFitSampleIdx - fitSamples
  
  np_data_early = np_data[firstFitSampleIdx:lastFitSampleIdx]
  
  startGuess_9kev = 3850
  startGuess_p6kev = 3450
  
  startGuess = startGuess_p6kev + (startGuess_9kev - startGuess_p6kev) * (energy -0.6)/9
  
  t0_guess = startGuess - firstFitSampleIdx
  startVal = startGuess
  baseline_guess = 0
  energy_guess = 1000 * energy / 5 #normalized for 5 keV i guess.  i dunno.  whatever.
  noise_sigma_guess = np.std(np_data[0:flatTimeSamples])
  
  iterations = 2000
  burnin = iterations-500
  #adaptiveDelay = 100
  
  
  #in case you gotta plot wtf is going on before the fit
#  plt.figure(1)
#  plt.clf()
#  #plt.title("Charge waveform")
#  plt.xlabel("Digitizer samples")
#  plt.ylabel("Raw ADC Value [Arb]")
#  plt.plot(np_data  ,color="red" )
#  value = raw_input('  --> Press q to quit, any other key to continue\n')
#  if value == 'q':
#    exit(1)

#  if doPlots:
#    verbosity = 1
#  else:
  verbosity = None

  
  siggen_model = pymc.Model( sm.createSignalModelSiggen(np_data_early, t0_guess, energy_guess, noise_sigma_guess, baseline_guess) )
  M = pymc.MCMC(siggen_model, verbose=verbosity)#, db="txt", dbname="Event_%d" % entryNumber)
  M.use_step_method(pymc.Metropolis, M.slowness_sigma, proposal_sd=1., proposal_distribution='Normal', verbose=verbosity)
  M.use_step_method(pymc.Metropolis, M.wfScale, proposal_sd=10., proposal_distribution='Normal', verbose=verbosity)
  M.use_step_method(pymc.DiscreteMetropolis, M.switchpoint, proposal_sd=1., proposal_distribution='Normal', verbose=verbosity)

#M.use_step_method(pymc.AdaptiveMetropolis, [M.slowness_sigma, M.wfScale, M.switchpoint], , shrink_if_necessary=1)
#  M.use_step_method(pymc.AdaptiveMetropolis, [M.radEst, M.zEst, M.phiEst, M.wfScale], delay=1000)
#  M.use_step_method(pymc.DiscreteMetropolis, M.switchpoint, proposal_distribution='Normal', proposal_sd=4)
  M.sample(iter=iterations, verbose=verbosity)

  t0 = np.around( np.median(M.trace('switchpoint')[burnin:]))
  scale =  np.median(M.trace('wfScale')[burnin:])
  sigma =  np.median(M.trace('slowness_sigma')[burnin:])
#  baselineB =  np.median(M.trace('baselineB')[burnin:])
#  baselineM =  0#np.median(M.trace('baselineM')[burnin:])
  startVal = t0 + firstFitSampleIdx
  
  sigma5 = np.percentile(M.trace('slowness_sigma')[burnin:], 5)
  sigma10 = np.percentile(M.trace('slowness_sigma')[burnin:], 10)

  #returnDict["spParam"] = sigma
#  print ">>> noise_sigma:    %f" % (M.trace('noise_sigma')[-1])**(-.5)

  M.halt()

  if doPlots:
    print ">>> startVal:    %d" % startVal
    print ">>> scale:       %f" % scale
    print ">>> slowness:    %f" % sigma
  
#########  Plots for MC Steps
    stepsFig = plt.figure()
    plt.clf()
    ax0 = stepsFig.add_subplot(311)
    ax1 = stepsFig.add_subplot(312, sharex=ax0)
    ax2 = stepsFig.add_subplot(313, sharex=ax0)
    
    ax0.plot(M.trace('switchpoint')[:])
    ax0.set_ylabel('t0')
    ax1.plot(M.trace('slowness_sigma')[:])
    ax1.set_ylabel('slowness')
    ax2.plot(M.trace('wfScale')[:])
    ax2.set_ylabel('energy')
    
  #  axarr[3].plot(M.trace('noise_sigma')[:])
  #  axarr[3].set_ylabel('noise_sigma')

    ax2.set_xlabel('MCMC Step Number')
    ax0.set_title('Raw MCMC Sampling')

    stepsFig.savefig(outputDir + "/energy%d/mcsteps_Event%d_energy%0.3f_spparam%0.3f.pdf" % (floor(energy), entryNumber, energy, sigma))

  #########  Waveform fit plot

    detZ = np.floor(30.0)/2.
    detRad = np.floor(30.3)
    phiAvg = np.pi/8
    siggen_fit = sm.findSiggenWaveform(detRad, phiAvg, detZ)
    siggen_fit *= scale

    out = np.zeros(len(np_data_early))
    out[t0:] += siggen_fit[0:(len(siggen_fit) - t0)]
    out = ndimage.filters.gaussian_filter1d(out, sigma)

    f = plt.figure()
    plt.clf()
    #plt.title("Charge waveform")
    plt.xlabel("Digitizer time [ns]")
    plt.ylabel("Raw ADC Value [Arb]")
    
    if waveletDenoise:
      plt.plot(np.arange(0, len(np_data)*10, 10), np_data_unfiltered  ,color="grey" )
    plt.plot(np.arange(0, len(np_data)*10, 10), np_data  ,color="red" )
    plt.xlim( firstFitSampleIdx*10, (lastFitSampleIdx+25)*10)

    plt.plot(np.arange(firstFitSampleIdx*10, lastFitSampleIdx*10, 10), out  ,color="blue" )
  #  plt.plot(np.arange(0, startVal), np.zeros(startVal)  ,color="blue" )
  #  plt.xlim( startVal-10, startVal+10)
  #  plt.ylim(-10, 25)
    plt.axvline(x=lastFitSampleIdx*10, linewidth=1, color='r',linestyle=":")
    plt.axvline(x=startVal*10, linewidth=1, color='g',linestyle=":")

    f.savefig(outputDir + "/energy%d/wf_Event%d_energy%0.3f_spparam%0.3f.pdf" % (floor(energy), entryNumber, energy, sigma))

    results_list.append( ((sigma, sigma5, sigma10), wfParams) )

#    value = raw_input('  --> Press q to quit, any other key to continue\n')
#    if value == 'q':
#      exit(1)

def setupDirs(outputDir, low, high):
  os.makedirs(outputDir)
  for i in xrange(low,high):
    os.makedirs(outputDir + "/energy%d" % i)

if __name__=="__main__":
    main(sys.argv[1:])


