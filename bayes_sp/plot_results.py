#!/usr/local/bin/python
import os, sys
import numpy as np
import matplotlib.pyplot as plt

def main(argv):

  (energy, spParam, risetime) = np.loadtxt("results.csv", skiprows=0, unpack=True, delimiter=",")


  fig = plt.figure(1)
  plt.scatter(energy, spParam)
  plt.xlabel("Energy [keV]")
  plt.ylabel("Fit Gaussian Sigma [samples]")

  fig2 = plt.figure(2)
  plt.scatter(risetime, spParam)
  plt.xlabel("10-90 risetime [4-6 keV range]")
  plt.ylabel("Fit Gaussian Sigma [samples]")

  plt.show()

if __name__=="__main__":
    main(sys.argv[1:])


