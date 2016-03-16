 #!/usr/local/bin/python

import numpy as np
import matplotlib.pyplot as plt


from pymc import Poisson, Normal




def createHistogramFitModel(data_bins, simulation_bins, scaleGuess):
  scale = Normal('scale', mu=scaleGuess, tau=sigToTau(.10*scaleGuess))
  scaled_sim = scale * simulation_bins

  baseline_observed = Poisson("baseline_observed", mu=scaled_sim, value=data_bins, observed= True )
  return locals()


################################################################################################################################

def sigToTau(sig):
  tau = np.power(sig, -2)
#  print "tau is %f" % tau
  return tau
################################################################################################################################
