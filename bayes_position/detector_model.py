#!/usr/local/bin/python
from ROOT import *
TROOT.gApplication.ExecuteFile("$MGDODIR/Root/LoadMGDOClasses.C")
TROOT.gApplication.ExecuteFile("$MGDODIR/Majorana/LoadMGDOMJClasses.C")
import numpy as np
import matplotlib.pyplot as plt
#Does all the interfacing with siggen for you, stores/loads lookup tables, and does electronics shaping

class Detector:
  def __init__(self, siggen_config_file, preampRisetime, preampFalltime, detZ, detRad ):
    #self.siggenConfigFile = siggen_config_file
    
    # in ns
    self.preampRiseTime = preampRisetime
    self.preampFallTime = preampFalltime
    
    # in mm
    self.length = detZ
    self.radius = detRad

    self.siggenSignalLength = 800
    self.zeroPadding = 200
    
    self.lookup_steps_r = 0.1
    self.lookup_steps_z = 0.1
    self.lookup_number_theta = 5
    self.lookupTable = None

#    self.lookup_steps_r = 0.1
#    self.lookup_steps_z = 0.1
#    self.lookup_number_theta = 25

    self.siggenInst = GATSiggenInstance(siggen_config_file)
#    self.rcint = MGWFRCIntegration()
#    self.rcdiff = MGWFRCDifferentiation()
#    self.rcint.SetTimeConstant(preampRisetime)
#    self.rcdiff.SetTimeConstant(preampFalltime)

########################################################################################################
  def GetSiggenWaveform(self, r,phi,z):

    x = r * np.sin(phi)
    y = r * np.cos(phi)

    #print "x, y, z is (%0.1f, %0.1f, %0.1f)" % (x,y,z)

    hitPosition = TVector3(x, y, z);
    sigWf = MGTWaveform();
    calcFlag = self.siggenInst.CalculateWaveform(hitPosition, sigWf, 1);
    
    if calcFlag == 0:
      siggen_data = sigWf.GetVectorData()
      siggen_data = np.multiply(siggen_data, 1) #changes it to numpy array
    
      print siggen_data
      print "Point out of crystal alert! (%0.3f,%0.3f,%0.3f)" % (r,phi,z)
      return None
    
    siggen_data = sigWf.GetVectorData()

    return siggen_data

########################################################################################################
  def GenerateLookupTable(self,fileName=None):
    r_range = np.arange(0, self.radius, self.lookup_steps_r)
    z_range = np.arange(0, self.length, self.lookup_steps_z)
    theta_range = np.linspace(0, np.pi/4, self.lookup_number_theta)

    lookupMatrix = np.empty([len(r_range), len(theta_range), len(z_range), self.siggenSignalLength])
  
    for (i_r,r) in enumerate(r_range):
      for (i_z,z) in enumerate(z_range):
        for (i_theta,theta) in enumerate(theta_range):
          print "Generating wf for (%f,%f,%f) w/ indices (%d,%d,%d)" % (r,theta,z, i_r,i_theta, i_z)
          wf = self.GetSiggenWaveform(r,theta,z)
          if wf is None:
            lookupMatrix[i_r,i_theta,i_z,:] = np.nan
          else:
            lookupMatrix[i_r,i_theta,i_z,:] = wf

    if fileName is not None:
      np.save(fileName, lookupMatrix)
    self.lookupTable = lookupMatrix

########################################################################################################
  def LoadLookupTable(self, fileName):
    self.lookupTable = np.load(fileName)
  
########################################################################################################
  def GetWaveformByPosition(self, r, theta, z):
    r_idx = np.around(r/self.lookup_steps_r)
    z_idx = np.around(z/self.lookup_steps_z)
    theta_steps = np.pi/4/(self.lookup_number_theta - 1)
    
    theta_idx = np.around(theta/theta_steps)
    
    print "--> lookup indices are (%d,%d,%d)" % (r_idx,theta_idx,z_idx)

    wf = self.lookupTable[r_idx, theta_idx, z_idx, :]
  
    if np.isnan(wf[0]):
      print "Out of crystal alert"
      return wf
    wf = np.pad(wf, (self.zeroPadding,0), 'constant', constant_values=(0, 0))
    
    plt.figure(1)
#    plt.clf()
    plt.plot(wf, color="red")
    plt.ylim(-0.1, 1.1)
    
    wf = self.RcDifferentiate( self.RcIntegrate(wf) )
    
    wf /= np.amax(wf)
    
    return wf
  
########################################################################################################

  def RcDifferentiate(self, anInput):
    timeConstantInSamples = self.preampFallTime / 10.
    dummy = anInput[0];
    anOutput = np.copy(anInput)
    dummy2 = 0.0;
    for i in xrange(1,len(anInput)):
      dummy2  = anOutput[i-1] + anInput[i] - dummy - anOutput[i-1] / timeConstantInSamples;
      dummy = anInput[i];
      anOutput[i] = dummy2;
   
    return anOutput

  def RcIntegrate(self, anInput, ):
    timeConstant= self.preampRiseTime/ 10. #switch to samples
    timeConstant = 1./timeConstant

    anOutput = np.copy(anInput)
    expTimeConstant = np.exp(-timeConstant);
    for i in xrange(1,len(anInput)):
      anOutput[i] = anInput[i] + expTimeConstant*anOutput[i-1];
    return anOutput







