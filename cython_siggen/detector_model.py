#!/usr/local/bin/python

#import sys
import numpy as np
import copy
from scipy import  signal, interpolate, ndimage

from siggen import Siggen

#Does all the interfacing with siggen for you, stores/loads lookup tables, and does electronics shaping

class Detector:
  def __init__(self, siggen_config_file, temperature=0, timeStep=None, numSteps=None, tfSystem = None):
  
    self.conf_file = siggen_config_file

    if timeStep is None or numSteps is None:
      self.siggenInst = Siggen(siggen_config_file)
    else:
      self.siggenInst =  Siggen(siggen_config_file, timeStep, numSteps)
      self.num_steps = np.int(numSteps)
      self.time_step_size = timeStep
    
      print "Time step size is %d" % self.time_step_size
      print "There will be %d steps in output" % self.num_steps
    
    (self.detector_radius, self.detector_length) = self.siggenInst.GetDimensions()
    
    self.SetTransferFunction(tfSystem)
  
    if temperature > 0:
      self.SetTemperature(temperature)

    #stuff for field interp
    self.wp_function = None
    self.efld_r_function = None
    self.efld_z_function = None
    self.rr = None
    self.zz = None
    self.wp_pp = None
    
    #stuff for waveform interpolation
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
    self.data_to_siggen_size_ratio = np.int(data_to_siggen_size_ratio)
    
    #Holders for wf simulation
    self.raw_siggen_data = np.zeros( self.num_steps, dtype=np.dtype('f4'), order="C" )
    self.processed_siggen_data = np.zeros( self.num_steps, dtype=np.dtype('f4'), order="C" )
    self.time_steps = np.arange(0, self.num_steps) * self.time_step_size*1E-9 #this is in ns
###########################################################################################################################
  def LoadFields(self, fieldFileName):
    self.fieldFileName = fieldFileName
  
    with np.load(fieldFileName) as data:
      data = np.load(fieldFileName)
      wpArray  = data['wpArray']
      efld_rArray = data['efld_rArray']
      efld_zArray = data['efld_zArray']
      gradList = data['gradList']
      pcRadList = data['pcRadList']
      pcLenList = data['pcLenList']
    
    self.gradList = gradList
    self.pcRadList = pcRadList
    self.pcLenList = pcLenList

    r_space = np.arange(0, wpArray.shape[0]/10. , 0.1, dtype=np.dtype('f4'))
    z_space = np.arange(0, wpArray.shape[1]/10. , 0.1, dtype=np.dtype('f4'))

    self.wp_function = interpolate.RegularGridInterpolator((r_space, z_space, pcRadList, pcLenList), wpArray)
    self.efld_r_function = interpolate.RegularGridInterpolator((r_space, z_space, gradList, pcRadList, pcLenList), efld_rArray)
    self.efld_z_function = interpolate.RegularGridInterpolator((r_space, z_space, gradList, pcRadList, pcLenList), efld_zArray)
    
    (self.rr, self.zz) = np.meshgrid(r_space, z_space)
###########################################################################################################################
  def SetFields(self, pcRad, pcLen, impurityGrad):
    self.pcRad = pcRad
    self.pcLen = pcLen
    self.impurityGrad = impurityGrad

    rr = self.rr
    zz = self.zz
    wp_function = self.wp_function
    efld_r_function = self.efld_r_function
    efld_z_function = self.efld_z_function
    
    radrad = np.ones_like(rr) * pcRad
    lenlen = np.ones_like(rr) * pcLen
    gradgrad = np.ones_like(rr) * impurityGrad
    
    points_wp =  np.array([rr.flatten() , zz.flatten(), radrad.flatten(), lenlen.flatten()], dtype=np.dtype('f4') ).T
    points_ef =  np.array([rr.flatten() , zz.flatten(), gradgrad.flatten(), radrad.flatten(), lenlen.flatten()], dtype=np.dtype('f4') ).T

    new_wp = np.array(wp_function( points_wp ).reshape(rr.shape).T, dtype=np.dtype('f4'), order="C")
    new_ef_r = np.array(efld_r_function( points_ef ).reshape(rr.shape).T, dtype=np.dtype('f4'), order="C")
    new_ef_z = np.array(efld_z_function( points_ef ).reshape(rr.shape).T, dtype=np.dtype('f4'), order="C")
  
    self.siggenInst.SetPointContact( pcRad, pcLen )
    self.siggenInst.SetFields(new_ef_r, new_ef_z, new_wp)
###########################################################################################################################
  def ReinitializeDetector(self):
    self.SetTemperature(self.temperature)
    self.SetFields(self.pcRad, self.pcLen, self.impurityGrad)
###########################################################################################################################
  def SetTemperature(self, temp):
    self.temperature = temp
    self.siggenInst.SetTemperature(temp)
###########################################################################################################################
  def SetTransferFunction(self, tfSystem):
    self.tfSystem = tfSystem
    tfSystem.to_tf()
    num = tfSystem.num
    den = tfSystem.den
    new_num, new_den, dt = signal.cont2discrete((num, den), 1E-8)
    self.num = new_num[0]
    self.den = new_den
###########################################################################################################################
  def IsInDetector(self, r, phi, z):
    if r > np.floor(self.detector_radius*10.)/10. or z > np.floor(self.detector_length*10.)/10.:
      return 0
    elif r <0 or z <0:
      return 0
    elif phi <0 or phi > np.pi/4:
      return 0
    elif r**2/self.pcRad**2 + z**2/self.pcLen**2 < 1:
      return 0
    else:
      return 1
###########################################################################################################################
  def GetSimWaveform(self, r,phi,z,scale, switchpoint,  numSamples, smoothing=None):
    sig_wf = self.GetRawSiggenWaveform(r, phi, z)
    if sig_wf is None:
      return None
    #smoothing for charge cloud size effects
    if smoothing is not None:
      ndimage.filters.gaussian_filter1d(sig_wf, smoothing, output=sig_wf)
    sim_wf = self.ProcessWaveform(sig_wf, numSamples, scale, switchpoint)
    return sim_wf
########################################################################################################
  def GetRawSiggenWaveform(self, r,phi,z, energy=1):

    x = r * np.sin(phi)
    y = r * np.cos(phi)
    
    self.raw_siggen_data.fill(0.)

    calcFlag = self.siggenInst.GetSignal(x, y, z, self.raw_siggen_data);
    if calcFlag == -1:
#      print "Holes out of crystal alert! (%0.3f,%0.3f,%0.3f)" % (r,phi,z)
      return None
    
    if np.amax(self.raw_siggen_data) == 0:
      print "found zero wf at r=%0.2f, phi=%0.2f, z=%0.2f (calcflag is %d)" % (r, phi, z, calcFlag)
      return None

    return self.raw_siggen_data
########################################################################################################
  def ProcessWaveform(self, siggen_wf, outputLength, scale, switchpoint):
    '''Use interpolation instead of rounding'''
    siggen_len = self.num_steps #+ self.zeroPadding
    
    switchpoint_ceil = switchpoint

    #actual wf gen
    siggen_wf= signal.lfilter(self.num, self.den, siggen_wf)
    smax = np.amax(siggen_wf)
    siggen_wf /= smax
    siggen_wf *= scale

    #resample the siggen wf to the 10ns digitized data frequency w/ interpolaiton
    switchpoint_ceil= np.int( np.ceil(switchpoint) )
    samples_to_fill = (outputLength - switchpoint_ceil)
    siggen_interp_fn = interpolate.interp1d(np.arange(self.num_steps ), siggen_wf, kind="linear", copy="False", assume_sorted="True")
    siggen_start_idx = (switchpoint_ceil - switchpoint) * self.data_to_siggen_size_ratio

    sampled_idxs = np.arange(samples_to_fill)*self.data_to_siggen_size_ratio + siggen_start_idx
    
    self.processed_siggen_data.fill(0.)

#    out[switchpoint_ceil:] = siggen_wf[:outputLength-switchpoint_ceil+1]
#    out *= scale

    try:
      self.processed_siggen_data[switchpoint_ceil:outputLength] = siggen_interp_fn(sampled_idxs)
    except ValueError:
      print "Something goofy happened here during interp"
      print outputLength
      print siggen_wf.size
      print sampled_idxs[-10:]
      return None
    return self.processed_siggen_data[:outputLength]
########################################################################################################
  #For pickling a detector object
  def __getstate__(self):
    # Copy the object's state from self.__dict__ which contains
    # all our instance attributes. Always use the dict.copy()
    # method to avoid modifying the original state.

    #manually do a deep copy of the velo data
    self.siggenSetup = self.siggenInst.GetSafeConfiguration()
  
    state = self.__dict__.copy()
    # Remove the unpicklable entries.
    del state['rr']
    del state['zz']
    del state['raw_siggen_data']
    del state['time_steps']
    del state['efld_r_function']
    del state['efld_z_function']
    del state['wp_function']
    del state['pcRadList']
    del state['gradList']
    del state['pcLenList']
    del state['siggenInst']

    return state

  def __setstate__(self, state):
    # Restore instance attributes
    self.__dict__.update(state)
    # Restore the previously opened file's state. To do so, we need to
    # reopen it and read from it until the line count is restored.

    self.siggenInst =  Siggen(savedConfig=self.siggenSetup)
  
    self.time_steps = np.arange(0, self.num_steps) * self.time_step_size*1E-9 #this is in ns
    self.raw_siggen_data = np.zeros( self.num_steps, dtype=np.dtype('f4'), order="C" )
    self.LoadFields(self.fieldFileName)
  

#  def __del__(self):
#    del self.wp_pp
#    del self.siggenInst

  