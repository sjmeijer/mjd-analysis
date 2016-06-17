#!/usr/local/bin/python
from ROOT import *

import numpy as np
from scipy import  signal, interpolate

#Does all the interfacing with siggen for you, stores/loads lookup tables, and does electronics shaping

class Detector:
  def __init__(self, siggen_config_file, temperature=0, timeStep=None, numSteps=None, tfSystem = None, zeroPadding=0):

    #TODO: GOTTA decide what to do about this
    self.zeroPadding = zeroPadding

    if timeStep is None or numSteps is None:
      self.siggenInst = GATSiggenInstance(siggen_config_file)
    else:
      self.siggenInst =  GATSiggenInstance(siggen_config_file, timeStep, numSteps)

    self.time_step_size = self.siggenInst.GetTimeStepLength()
    self.num_steps = self.siggenInst.GetTimeStepNumber()
    self.detector_radius = self.siggenInst.GetDetectorRadius()
    self.detector_length = self.siggenInst.GetDetectorLength()
    
    self.time_steps = np.arange(0, self.num_steps+ self.zeroPadding) * self.time_step_size*1E-9 #this is in ns
    
    self.tfSystem = tfSystem
  
    if temperature > 0:
      self.siggenInst.SetTemperature(temperature)

  def IsInDetector(self, r, phi, z):
    if r > self.detector_radius or z > self.detector_length:
      return 0
    elif r <0 or z <0:
      return 0
    elif phi <0 or phi > np.pi/4:
      return 0
    else:
      return 1


  def GetSimWaveform(self, r,phi,z,scale, switchpoint,  numSamples, temp=None, num=None, den=None):
  
    if num is not None and den is not None:
      self.tfSystem = signal.lti(num, den)
    
    if temp is not None:
      self.siggenInst.SetTemperature(temp)
    
  
    sig_wf = self.GetRawSiggenWaveform(r, phi, z)
    
    if sig_wf is None:
      return None
    
    sim_wf = self.ProcessWaveform(sig_wf, numSamples, scale, switchpoint)
    
    return sim_wf

########################################################################################################
  def GetRawSiggenWaveform(self, r,phi,z, energy=1):

    x = r * np.sin(phi)
    y = r * np.cos(phi)

    hitPosition = TVector3(x, y, z);
    
    sigWf = MGTWaveform();
    calcFlag = self.siggenInst.CalculateWaveform(hitPosition, sigWf, energy);
    if calcFlag == 0:
      print "Holes out of crystal alert! (%0.3f,%0.3f,%0.3f)" % (r,phi,z)
      return None
    siggen_data = np.multiply(sigWf.GetVectorData(),1)

    return siggen_data

########################################################################################################

  def ProcessWaveform(self, siggen_wf, outputLength, scale, switchpoint):
    '''Use interpolation instead of rounding'''
  
    siggen_len = self.num_steps + self.zeroPadding


    #TODO: i don't think zero padding actually does anything anyway
    if self.zeroPadding == 0:
      zeroPaddingIdx = 0
    else:
      siggen_wf = np.pad(siggen_wf, (self.zeroPadding,0), 'constant', constant_values=(0, 0))
      zeroPaddingIdx = self.zeroPadding
    
    #actual wf gen
    tout, siggen_wf, x = signal.lsim(self.tfSystem, siggen_wf, self.time_steps)
    siggen_wf /= np.amax(siggen_wf)
    
    siggen_data = siggen_wf[zeroPaddingIdx::]
    siggen_data = siggen_data*scale
    

    #I think most this stuff could be shoved to init to avoid having to redo it on the fly
    
    #round here to fix floating point accuracy problem
    data_to_siggen_size_ratio = np.around(10. / self.time_step_size,3)
    
    if not data_to_siggen_size_ratio.is_integer():
      print "Error: siggen step size must evenly divide into 10 ns digitization period (ratio is %f)" % data_to_siggen_size_ratio
      exit(0)
    elif data_to_siggen_size_ratio < 10:
      round_places = 0
    elif data_to_siggen_size_ratio < 100:
      round_places = 1
    elif data_to_siggen_size_ratio < 1000:
      round_places = 2
    else:
      print "Error: Ben was too lazy to code in support for resolution this high"
      exit(0)
    
    data_to_siggen_size_ratio = np.int(data_to_siggen_size_ratio)


    #resample the siggen wf to the 10ns digitized data frequency w/ interpolaiton
    switchpoint_ceil= np.int( np.ceil(switchpoint) )
    
    
    samples_to_fill = (outputLength - switchpoint_ceil)
    
#    print "num steps %d" % self.num_steps
#    print "len siggen data %d" % len(siggen_data)

    siggen_interp_fn = interpolate.interp1d(np.arange(self.num_steps ), siggen_data, kind="linear")

    siggen_start_idx = switchpoint_ceil - switchpoint
    
#    print "siggen start idx is %f" % siggen_start_idx

    sampled_idxs = np.arange(samples_to_fill)*data_to_siggen_size_ratio + siggen_start_idx
    
#    print sampled_idxs

    out = np.zeros(outputLength)
    out[switchpoint_ceil:] = siggen_interp_fn(sampled_idxs)
    return out


#  def ProcessWaveform(self, siggen_wf, outputLength, scale, switchpoint):
#  
#    siggen_len = self.num_steps + self.zeroPadding
#
#    #actual wf gen
#    tout, siggen_wf, x = signal.lsim(self.tfSystem, siggen_wf, self.time_steps)
#    siggen_wf /= np.amax(siggen_wf)
#    
#    #TODO: by doing this, I have no idea what I'm trying to accomplish with zero padding
#    siggen_data = siggen_wf[self.zeroPadding::]
#    siggen_data = siggen_data*scale
#    
#
#    #I think most this stuff could be shoved to init to avoid having to redo it on the fly
#    
#    #round here to fix floating point accuracy problem
#    data_to_siggen_size_ratio = np.around(10. / self.time_step_size,3)
#    
#    if not data_to_siggen_size_ratio.is_integer():
#      print "Error: siggen step size must evenly divide into 10 ns digitization period (ratio is %f)" % data_to_siggen_size_ratio
#      exit(0)
#    elif data_to_siggen_size_ratio < 10:
#      round_places = 0
#    elif data_to_siggen_size_ratio < 100:
#      round_places = 1
#    elif data_to_siggen_size_ratio < 1000:
#      round_places = 2
#    else:
#      print "Error: Ben was too lazy to code in support for resolution this high"
#      exit(0)
#    
#    data_to_siggen_size_ratio = np.int(data_to_siggen_size_ratio)
#
#
#    #resample the siggen wf to the 10ns digitized data frequency
#    siggen_start_idx = np.int(np.around(switchpoint, decimals=round_places) * data_to_siggen_size_ratio % data_to_siggen_size_ratio)
#    switchpoint_ceil= np.int( np.ceil(switchpoint) )
#    samples_to_fill = (outputLength - switchpoint_ceil)
#    sampled_idxs = np.arange(samples_to_fill, dtype=np.int)*data_to_siggen_size_ratio + siggen_start_idx
#
#    if 0:
#      print "siggen step size: %f" % self.time_step_size
#      print "data to siggen ratio: %f" % data_to_siggen_size_ratio
#      print "switchpoint: %f" % switchpoint
#      print "siggen start idx: %d" % siggen_start_idx
#      print "switchpoint ceil: %d" % switchpoint_ceil
#      print "final idx: %d" % ((len(data) - switchpoint_ceil)*10)
#      print "samples to fill: %d" % samples_to_fill
#      print sampled_idxs
#      print siggen_data[sampled_idxs]
#    
#    out = np.zeros(outputLength)
#    out[switchpoint_ceil:] = siggen_data[sampled_idxs]
#    return out


  def SetTemperature(self, temp):
    self.siggenInst.SetTemperature(temp)







