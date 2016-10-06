#!/usr/local/bin/python
import matplotlib
#matplotlib.use('CocoaAgg')
import sys, os
import matplotlib.pyplot as plt
import numpy as np
from detector_model import *
from matplotlib.colors import LogNorm

import helpers

def main(argv):


  timeStepSize = 1
  
  wfFileName = "most_recent_wf_pt.npz"
  numWaveforms = 30
  #wfFileName = "P42574A_512waveforms_%drisetimeculled.npz" % numWaveforms
  if os.path.isfile(wfFileName):
    data = np.load(wfFileName)
    samples = data['samples']
  else:
    print "No saved waveform available."
    
  tempGuess = 79.204603
  gradGuess = 0.05
  pcRadGuess = 2.5
  pcLenGuess = 1.6

  #Create a detector model
  detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.05,2.5, 1.65)
  det =  Detector(detName, temperature=tempGuess, timeStep=timeStepSize, numSteps=fitSamples*10./timeStepSize,)
  det.LoadFields("P42574A_fields_v3.npz")
  det.SetFields(pcRadGuess, pcLenGuess, gradGuess)

  b_over_a = 0.107077
  c = -0.817381
  d = 0.825026
  rc = 76.551780
  det.SetTransferFunction(b_over_a, c, d, rc)

  positionFig = plt.figure(5)
  plt.clf()
  xedges = np.linspace(0, np.around(det.detector_radius,1), np.around(det.detector_radius,1)*50+1)
  yedges = np.linspace(0, np.around(det.detector_length,1), np.around(det.detector_length,1)*50+1)
  z, xe, ye = np.histogram2d(samples[:,0], samples[:,2],   bins=[ xedges,yedges  ])
  
  z /= z.sum()
  n=100
  t = np.linspace(0, z.max(), n)
  integral = ((z >= t[:, None, None]) * z).sum(axis=(1,2))
  from scipy import interpolate
  f = interpolate.interp1d(integral, t)
  t_contours = f(np.array([0.95, .68, .5, .2]))

  cs = plt.contourf(z.T, t_contours, extent=[0,det.detector_radius,0,det.detector_length],  alpha=0.7, colors = ("green","r","c", "b"), extend='max')
  cs.cmap.set_over('red')
  
  proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) 
    for pc in cs.collections]

  plt.legend(proxy, ["95% C.I.", "68% C.I.", "50% C.I.", "20% C.I."], loc=3)
  plt.xlabel("r from Point Contact (mm)")
  plt.ylabel("z from Point Contact (mm)")
  
  plt.xlim(8,14)
  plt.ylim(14,21)

  plt.savefig("pt_credible_intervals.pdf")

  print len(np.where( z >= t_contours[-1]  )[0]) * .02*.02

  plt.figure()
  plt.imshow(z.T, origin='lower', extent=[0,det.detector_radius,0,det.detector_length],  norm=LogNorm())
  plt.xlabel("r from Point Contact (mm)")
  plt.ylabel("z from Point Contact (mm)")

  plt.savefig("pt_heat_map.pdf")


  plt.figure()
  wfPlotNumber = 100
  simWfs = np.empty((wfPlotNumber, fitSamples) )

  for idx, (r, phi, z, scale, t0, smooth, ) in enumerate(samples[np.random.randint(len(samples), size=wfPlotNumber)]):
    simWfs[idx,:] = det.MakeSimWaveform(r, phi, z, scale, t0, fitSamples, h_smoothing = smooth,)

  residFig = plt.figure(3)
  helpers.plotResidual(simWfs, wf.windowedWf, figure=residFig)



  plt.show()



if __name__=="__main__":
    main(sys.argv[1:])


