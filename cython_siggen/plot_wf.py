#!/usr/local/bin/python
import matplotlib
#matplotlib.use('CocoaAgg')
import sys, os
import matplotlib.pyplot as plt
import numpy as np
from detector_model import *
from matplotlib.colors import LogNorm

def main(argv):


  timeStepSize = 1
  
  wfFileName = "most_recent_wf.npz"
  numWaveforms = 30
  #wfFileName = "P42574A_512waveforms_%drisetimeculled.npz" % numWaveforms
  if os.path.isfile(wfFileName):
    data = np.load(wfFileName)
    samples = data['samples']
  else:
    print "No saved waveform available."
    
  detName = "conf/P42574A_grad%0.2f_pcrad%0.2f_pclen%0.2f.conf" % (0.05,2.5, 1.65)
  det =  Detector(detName,timeStep=1, numSteps=1000)

  positionFig = plt.figure(5)
  plt.clf()
  xedges = np.linspace(0, np.around(det.detector_radius,1), np.around(det.detector_radius,1)*50+1)
  yedges = np.linspace(0, np.around(det.detector_length,1), np.around(det.detector_length,1)*50+1)
  z, xe, ye = np.histogram2d(samples[:,0], samples[:,2],   bins=[ xedges,yedges  ])
  
  z /= z.sum()
  n=1000
  t = np.linspace(0, z.max(), n)
  integral = ((z >= t[:, None, None]) * z).sum(axis=(1,2))
  from scipy import interpolate
  f = interpolate.interp1d(integral, t)
  t_contours = f(np.array([.68, .95, .9973]))
  plt.imshow(z.T, origin='lower', extent=[0,det.detector_radius,0,det.detector_length],  norm=LogNorm(),cmap="Greys")
#    plt.colorbar()

  print t_contours

  CS3 = plt.contour(z.T, t_contours, extent=[0,det.detector_radius,0,det.detector_length],   colors = ("r","b", "g"))
#  CS3.cmap.set_over('red')

  plt.xlabel("r from Point Contact (mm)")
  plt.ylabel("z from Point Contact (mm)")

  print len(np.where( z >= t_contours[0]  )[0]) * .02*.02

  


  plt.show()



if __name__=="__main__":
    main(sys.argv[1:])


