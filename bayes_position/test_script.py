#!/usr/local/bin/python
from ROOT import *
import numpy as np
import matplotlib.pyplot as plt

from detector_model import *

det = Detector("P42661C_autogen_final.conf", 190 * CLHEP.ns, 69.88*CLHEP.us, 41.5, 35.4)
det.GenerateLookupTable("P42661C.npy")
#det.LoadLookupTable("P42661C.npy")
r_range = np.arange(0, det.radius, det.lookup_steps_r)
z_range = np.arange(0, det.length, det.lookup_steps_z)
theta_range = np.linspace(0, np.pi/4, det.lookup_number_theta)

plt.ion()
f1=plt.figure(1)
f2=plt.figure(2)

for i in r_range:
  for j in theta_range:
    for k in z_range:
    
      print "Looking for (%f,%f,%f)" % (i,j,k)
      wf = det.GetWaveformByPosition(i,j,k)
      plt.figure(2)
#      plt.clf()
      plt.plot(wf, color="blue")
      plt.ylim(-0.1, 1.1)

  value = raw_input('  --> Press q to quit, any other key to continue\n')
  if value == 'q':
    exit(1)