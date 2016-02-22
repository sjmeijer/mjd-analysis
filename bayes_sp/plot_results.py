#!/usr/local/bin/python
import os, sys
import numpy as np
import matplotlib.pyplot as plt

from ROOT import *

def main(argv):

  wf_file = TFile('spParamSkim.root')
  tree_wf = wf_file.Get('slowpulseParamTree')
  
  energy = []
  spParam = []
  wp = []
  
  for i in xrange(tree_wf.GetEntries()):
    tree_wf.GetEntry(i)
    energy.append(tree_wf.energykeV)
    spParam.append(tree_wf.spParam)
    wp.append(tree_wf.wpar)
  

  fig = plt.figure(1)
  plt.scatter(energy, spParam, s=2)
  plt.xlabel("Energy [keV]")
  plt.ylabel("Fit Gaussian Sigma [samples]")
  plt.xlim(0.5,10)
  plt.ylim(0, np.amax(spParam))

  fig.savefig("bayes.png")

  plt.ylim(0, 100)
  fig.savefig("bayes_zoom.png")

  fig = plt.figure(2)
  plt.scatter(energy, wp, s=2)
  plt.xlabel("Energy [keV]")
  plt.ylabel("Wpar")
  plt.xlim(0.5,10)
  plt.ylim(0, np.amax(wp))

  fig.savefig("wpar.png")

  plt.ylim(0, 5000)
  fig.savefig("wpar_zoom.png")



  plt.show()

if __name__=="__main__":
    main(sys.argv[1:])


