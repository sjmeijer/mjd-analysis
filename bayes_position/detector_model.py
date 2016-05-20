#!/usr/local/bin/python
from ROOT import *
TROOT.gApplication.ExecuteFile("$MGDODIR/Root/LoadMGDOClasses.C")
TROOT.gApplication.ExecuteFile("$MGDODIR/Majorana/LoadMGDOMJClasses.C")
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, signal

#Does all the interfacing with siggen for you, stores/loads lookup tables, and does electronics shaping

class Detector:
  def __init__(self, siggen_config_file, detZ, detRad,  preampRisetime, preampFalltimeLong, preampFalltimeShort=0,  preampFalltimeShortFraction=0, zeroPadding=200, chargeTrappingTime = 0, gaussian_smoothing = 0, temperature=0, preampRC = 0, preampLC = 0):
    #self.siggenConfigFile = siggen_config_file
    
    # in ns
    self.preampRiseTime = preampRisetime
    self.preampFalltimeLong = preampFalltimeLong
    self.preampFalltimeShort = preampFalltimeShort
    self.preampFalltimeShortFraction = preampFalltimeShortFraction
    self.chargeTrappingTime = chargeTrappingTime
    
    
    #RLC Circuit
    self.preampRC = preampRC
    self.preampLC = preampLC
    
    
    # in mm
    self.length = detZ
    self.radius = detRad

    self.siggenSignalLength = 800
    self.zeroPadding = zeroPadding
    
    self.lookup_steps_r = 1. #mm
    self.lookup_steps_z = 1. #mm
    self.lookup_number_theta = 5
    self.lookupTable = None
    
    self.gaussian_smoothing = gaussian_smoothing



#    self.lookup_steps_r = 0.1
#    self.lookup_steps_z = 0.1
#    self.lookup_number_theta = 25

    self.siggenInst = GATSiggenInstance(siggen_config_file)
  
  
    if temperature > 0:
      self.siggenInst.SetTemperature(temperature)


########################################################################################################
  def GetSiggenWaveform(self, r,phi,z, energy=1):

    x = r * np.sin(phi)
    y = r * np.cos(phi)

    #print "x, y, z is (%0.1f, %0.1f, %0.1f)" % (x,y,z)

    hitPosition = TVector3(x, y, z);
    
#    sigWfHoles = MGTWaveform();
#    calcFlagHoles = self.siggenInst.MakeSignal(hitPosition, sigWfHoles, 1., energy);
#    if calcFlagHoles == 0:
#      print "Holes out of crystal alert! (%0.3f,%0.3f,%0.3f)" % (r,phi,z)
#      return None
#    
#    sigWfElectrons = MGTWaveform();
#    calcFlagElectrons = self.siggenInst.MakeSignal(hitPosition, sigWfElectrons, -1.,energy);
#    if calcFlagElectrons == 0:
#      print "Holes out of crystal alert! (%0.3f,%0.3f,%0.3f)" % (r,phi,z)
#      return None
#
#    siggen_data_holes = sigWfHoles.GetVectorData()
#    siggen_data_electrons = sigWfElectrons.GetVectorData()
#    
#    siggen_data = np.add(siggen_data_holes, siggen_data_electrons)

    sigWf = MGTWaveform();
    calcFlag = self.siggenInst.CalculateWaveform(hitPosition, sigWf, energy);
    if calcFlag == 0:
#      siggen_data = sigWf.GetVectorData()
#      siggen_data = np.multiply(siggen_data, 1) #changes it to numpy array
#    
#      print siggen_data
      print "Holes out of crystal alert! (%0.3f,%0.3f,%0.3f)" % (r,phi,z)
      return None
    siggen_data = np.multiply(sigWf.GetVectorData(),1)

    return siggen_data
    
  def ProcessWaveform(self, wf):
  
    wf = np.pad(wf, (self.zeroPadding,0), 'constant', constant_values=(0, 0))
    

    
    if self.chargeTrappingTime > 0:
      wf = self.RcDifferentiate(wf, self.chargeTrappingTime)
    
    rc_int = self.RcIntegrate(wf)
    
    siggen_data_pz_long = self.RcDifferentiate(rc_int, self.preampFalltimeLong)
    
    
    if self.preampFalltimeShort > 0:
#      plt.figure()
#      plt.plot(rc_int, color="r")
#      plt.plot(siggen_data_pz_long, color="b")
      siggen_data_pz_short = self.RcDifferentiate(rc_int, self.preampFalltimeShort)
#      plt.plot(siggen_data_pz_short, color="g")

      wf = (1-self.preampFalltimeShortFraction)*siggen_data_pz_long +self.preampFalltimeShortFraction*siggen_data_pz_short

    else:
      wf = siggen_data_pz_long
    
    if self.gaussian_smoothing > 0:
      wf = ndimage.filters.gaussian_filter1d(wf, self.gaussian_smoothing)
    
    wf /= np.amax(wf)
    

    return wf
    
  def ProcessWaveformRLC(self, wf):
  
    #zero pad at start to allow for smearing
    wf = np.pad(wf, (self.zeroPadding,0), 'constant', constant_values=(0, 0))
    

    num = [1]
    den = [self.preampLC, self.preampRC, 1]
    system = signal.TransferFunction(num, den)
    t = np.arange(0, len(wf))
    tout, wf, x = signal.lsim(system, wf, t)

    wf = self.RcDifferentiate(wf, self.preampFalltimeLong)
    
    wf /= np.amax(wf)
    

    return wf
  
  def ProcessWaveformOpAmp(self, wf):
  
    #zero pad at start to allow for smearing
    wf = np.pad(wf, (self.zeroPadding,0), 'constant', constant_values=(0, 0))
    

    num = [self.preampR2]
    den = [self.preampR1*self.preampR2*self.preampC, self.preampR1]
    system = signal.TransferFunction(num, den)
    t = np.arange(0, len(wf))
    tout, wf, x = signal.lsim(system, wf, t)

    wf = self.RcDifferentiate(wf, self.preampFalltimeLong)
    
    wf /= np.amax(wf)
    

    return wf
  
  def ProcessWaveformFoldedCascode(self, wf):
  
    #zero pad at start to allow for smearing
    wf = np.pad(wf, (self.zeroPadding,0), 'constant', constant_values=(0, 0))
    
    num = [self.preampR2]
    den = [self.preampR1*self.preampR2*self.preampC, self.preampR1]
    system = signal.TransferFunction(num, den)
    t = np.arange(0, len(wf))
    tout, wf, x = signal.lsim(system, wf, t)
    
    num2 = [self.fc_C1*self.fc_R,0]
    den2 = [self.fc_C2*self.fc_R, 1]
    system2 = signal.TransferFunction(num2, den2)
    tout, wf, x = signal.lsim(system2, wf, t)

#    wf = self.RcDifferentiate(wf, self.preampFalltimeLong)

    wf /= np.amax(wf)
    

    return wf


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
    print "Loading lookup table.  Could take a while.  Throw on some Mac and chill a minute."
    self.lookupTable = np.load(fileName)
  
########################################################################################################
  def GetWaveformByPosition(self, r, theta, z):
    r_idx = np.around(r/self.lookup_steps_r)
    z_idx = np.around(z/self.lookup_steps_z)
    theta_steps = np.pi/4/(self.lookup_number_theta - 1)
    
    theta_idx = np.around(theta/theta_steps)
    
#    print "--> lookup position is (%0.2f,%0.2f,%0.2f) indices (%d,%d,%d)" % (r,theta,z, r_idx,theta_idx,z_idx)

    wf = self.lookupTable[r_idx, theta_idx, z_idx, :]
  
    if np.isnan(wf[0]):
      print "Out of crystal alert"
      return wf
#    wf = self.ProcessWaveform(wf)

    
#    plt.figure(1)
#    plt.clf()
#    plt.plot(wf[self.zeroPadding:self.zeroPadding+110], color="blue")
#    plt.ylim(-0.1, 1.1)

    #kills some of the zero padding.  makes it easier for me to deal with for now.
    return wf[self.zeroPadding:]
    
  def GetWaveformByIndex(self, rIdx, phiIdx, zIdx):
    wf = self.lookupTable[rIdx, phiIdx, zIdx, :]
  
    if np.isnan(wf[0]):
      print "Out of crystal alert"
      return None

    return wf
  
  
########################################################################################################

  def RcDifferentiate(self, anInput, timeConstantInNs = None):
    if timeConstantInNs is None:
      timeConstantInNs = self.preampFallTime
    timeConstantInSamples = timeConstantInNs / 10.
    dummy = anInput[0];
    anOutput = np.copy(anInput)
    dummy2 = 0.0;
    for i in xrange(1,len(anInput)):
      dummy2  = anOutput[i-1] + anInput[i] - dummy - anOutput[i-1] / timeConstantInSamples;
      dummy = anInput[i];
      anOutput[i] = dummy2;
   
    return anOutput

  def RcIntegrate(self, anInput ):
    timeConstant= self.preampRiseTime/ 10. #switch to samples
    timeConstant = 1./timeConstant

    anOutput = np.copy(anInput)
    expTimeConstant = np.exp(-timeConstant);
    for i in xrange(1,len(anInput)):
      anOutput[i] = anInput[i] + expTimeConstant*anOutput[i-1];
    return anOutput

  def SetTemperature(self, temp):
    self.siggenInst.SetTemperature(temp)







