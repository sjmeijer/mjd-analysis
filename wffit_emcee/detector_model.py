#!/usr/local/bin/python
from ROOT import *

import numpy as np
import copy
from scipy import  signal, interpolate

#don't tell me you don't know what a float** is
import warnings
warnings.filterwarnings( action='ignore', category=RuntimeWarning, message='creating converter.*' )


#Does all the interfacing with siggen for you, stores/loads lookup tables, and does electronics shaping

class Detector:
  def __init__(self, siggen_config_file, temperature=0, timeStep=None, numSteps=None, tfSystem = None, zeroPadding=0):
  
    self.conf_file = siggen_config_file

    #TODO: GOTTA decide what to do about this
    self.zeroPadding = zeroPadding

    if timeStep is None or numSteps is None:
      self.siggenInst = GATSiggenInstance(siggen_config_file)
    else:
      self.siggenInst =  GATSiggenInstance(siggen_config_file, timeStep, numSteps)

    self.time_step_size = self.siggenInst.GetTimeStepLength()
    self.num_steps = np.int( self.siggenInst.GetTimeStepNumber() )
    self.detector_radius = self.siggenInst.GetDetectorRadius()
    self.detector_length = self.siggenInst.GetDetectorLength()
    
    self.time_steps = np.arange(0, self.num_steps+ self.zeroPadding) * self.time_step_size*1E-9 #this is in ns
    
    self.tfSystem = tfSystem
  
    if temperature > 0:
      self.siggenInst.SetTemperature(temperature)

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
    self.sigWf = MGTWaveform();
    self.raw_siggen_data = np.empty( self.num_steps )
#    self.processed_sim_data = np.empty()

      
  def LoadFields(self, fieldFileName):
  
    with np.load(fieldFileName) as data:
      data = np.load(fieldFileName)
      wpArray  = data['wpArray']
      efld_rArray = data['efld_rArray']
      efld_zArray = data['efld_zArray']
      gradList = data['gradList']
      pcRadList = data['pcRadList']
    
    self.gradList = gradList
    self.pcRadList = pcRadList
    
#    print "gradList is " + str(gradList)
#    print "pcRadList is " + str(pcRadList)


    r_space = np.arange(0, wpArray.shape[0]/10. , 0.1, dtype=np.dtype('f4'))
    z_space = np.arange(0, wpArray.shape[1]/10. , 0.1, dtype=np.dtype('f4'))

    self.wp_function = interpolate.RegularGridInterpolator((r_space, z_space, pcRadList), wpArray)
    self.efld_r_function = interpolate.RegularGridInterpolator((r_space, z_space, gradList, pcRadList), efld_rArray)
    self.efld_z_function = interpolate.RegularGridInterpolator((r_space, z_space, gradList, pcRadList), efld_zArray)
    
    (self.rr, self.zz) = np.meshgrid(r_space, z_space)


  def SetFields(self, pcSize, impurityGrad):
#    print "setting pc radius to %0.4f, grad to %0.4f" % (pcSize, impurityGrad)

    rr = self.rr
    zz = self.zz
    wp_function = self.wp_function
    efld_r_function = self.efld_r_function
    efld_z_function = self.efld_z_function
    
    pcpc = np.ones_like(rr) * pcSize
    gradgrad = np.ones_like(rr) * impurityGrad
    
    points_wp =  np.array([rr.flatten() , zz.flatten(), pcpc.flatten()], dtype=np.dtype('f4') ).T
    points_ef =  np.array([rr.flatten() , zz.flatten(), gradgrad.flatten(), pcpc.flatten()], dtype=np.dtype('f4') ).T
    
    
    new_wp = np.array(wp_function( points_wp ).reshape(rr.shape).T, dtype=np.dtype('f4'), order="C")
    new_ef_r = np.array(efld_r_function( points_ef ).reshape(rr.shape).T, dtype=np.dtype('f4'), order="C")
    new_ef_z = np.array(efld_z_function( points_ef ).reshape(rr.shape).T, dtype=np.dtype('f4'), order="C")
  
    self.wp_pp = getPointer(new_wp)
    efr_pp = getPointer(new_ef_r)
    efz_pp = getPointer(new_ef_z)
  
    self.siggenInst.SetWeightingPotential( self.wp_pp )
    self.siggenInst.SetElectricField( efr_pp, efz_pp )

  def IsInDetector(self, r, phi, z):
    if r > self.detector_radius or z > self.detector_length:
      return 0
    elif r <0 or z <0:
      return 0
#    elif phi <0 or phi > np.pi/4:
#      return 0
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
    
    calcFlag = self.siggenInst.CalculateWaveform(hitPosition, self.sigWf, energy);
    if calcFlag == 0:
      print "Holes out of crystal alert! (%0.3f,%0.3f,%0.3f)" % (r,phi,z)
      return None
    self.raw_siggen_data = np.array(self.sigWf.GetVectorData(),copy=False)

    return self.raw_siggen_data

########################################################################################################

  def ProcessWaveform(self, siggen_wf, outputLength, scale, switchpoint):
    '''Use interpolation instead of rounding'''
  
    siggen_len = self.num_steps #+ self.zeroPadding

#    #TODO: i don't think zero padding actually does anything anyway
#    if self.zeroPadding == 0:
#      zeroPaddingIdx = 0
#    else:
#      siggen_wf = np.pad(siggen_wf, (self.zeroPadding,0), 'constant', constant_values=(0, 0))
#      zeroPaddingIdx = self.zeroPadding

    #actual wf gen
    tout, siggen_data, x = signal.lsim(self.tfSystem, siggen_wf, self.time_steps)
    siggen_data /= np.amax(siggen_data)
    
#    siggen_data = siggen_wf[zeroPaddingIdx::]
    siggen_data *= scale

    #resample the siggen wf to the 10ns digitized data frequency w/ interpolaiton
    switchpoint_ceil= np.int( np.ceil(switchpoint) )
    samples_to_fill = (outputLength - switchpoint_ceil)


    siggen_interp_fn = interpolate.interp1d(np.arange(self.num_steps ), siggen_data, kind="linear", copy="False", assume_sorted="True")

    siggen_start_idx = switchpoint_ceil - switchpoint
    

    sampled_idxs = np.arange(samples_to_fill)*self.data_to_siggen_size_ratio + siggen_start_idx
    
#    print sampled_idxs

    out = np.zeros(outputLength)
    out[switchpoint_ceil:] = siggen_interp_fn(sampled_idxs)
    return out


  def SetTransferFunction(self, num, den):
    self.tfSystem = signal.lti(num, den)

  def SetTemperature(self, temp):
    self.siggenInst.SetTemperature(temp)

  def PlotFields(self):
    import matplotlib.pyplot as plt
  
    det = self

    wp = np.array(det.siggenInst.GetWeightingPotential(), dtype=np.dtype('f4'), order='C')

    plt.figure()
    
  #  wp[np.where(wp==0)] = 1

    plt.imshow(wp.T, origin='lower',  interpolation='nearest', cmap=plt.cm.RdYlBu_r)
    
    plt.title("WP from siggen memory")
    plt.xlabel("radial (mm)")
    plt.ylabel("axial (mm)")
    
  #  plt.xlim(0,5)
  #  plt.ylim(0,5)

    det.siggenInst.ReadElectricField()
    efld_r = np.array(det.siggenInst.GetElectricFieldR(), dtype=np.dtype('f4'), order='C')
    efld_phi = np.array(det.siggenInst.GetElectricFieldPhi(), dtype=np.dtype('f4'), order='C')
    
    if not np.array_equal(efld_phi, np.zeros_like(efld_phi)):
        print "WARNING: PHI NOT ALL ZERO!"
    
    efld_z = np.array(det.siggenInst.GetElectricFieldZ(), dtype=np.dtype('f4'), order='C')

    plt.figure()
    mag = np.sqrt( np.add(np.add(np.square(efld_r), np.square(efld_phi)), np.square(efld_z))  )

  #  mag[np.where(mag==0)] = np.nan

    plt.imshow(mag.T, origin='lower',  interpolation='nearest', cmap=plt.cm.RdYlBu_r)
    
    plt.title("E field from siggen memory")
    plt.xlabel("radial (mm)")
    plt.ylabel("axial (mm)")


  #For pickling a detector object
  def __getstate__(self):
    # Copy the object's state from self.__dict__ which contains
    # all our instance attributes. Always use the dict.copy()
    # method to avoid modifying the original state.
    
#    self.siggenSetup = Safe_Siggen_Setup()
#    self.siggenInst.SaveSiggenSetup(self.siggenSetup)

    #manually do a deep copy of the velo data
    self.siggenSetup = self.siggenInst.GetSafeSiggenSetup()
    self.siggenVelo = v_lookup(self.siggenSetup.velo_data)
  
    state = self.__dict__.copy()
    # Remove the unpicklable entries.
    del state['wp_pp']
    del state['siggenInst']
    state['wp_pp'] = None
    return state

  def __setstate__(self, state):
    # Restore instance attributes (i.e., filename and lineno).
    self.__dict__.update(state)
    # Restore the previously opened file's state. To do so, we need to
    # reopen it and read from it until the line count is restored.

    self.siggenSetup.velo_data = self.siggenVelo.v_lookup_obj
    self.siggenInst =  GATSiggenInstance(self.siggenSetup)

  def __del__(self):
    del self.wp_pp
    del self.siggenInst

def getPointer(floatfloat):
  return (floatfloat.__array_interface__['data'][0] + np.arange(floatfloat.shape[0])*floatfloat.strides[0]).astype(np.intp)


#wrap the safe_siggen_setup for pickling
class v_lookup:
  def __init__(self, v_lookup_obj):
    self.v_lookup_obj = v_lookup_obj
    self.e = v_lookup_obj.e
    self.e100 = v_lookup_obj.e100
    self.e110 = v_lookup_obj.e110
    self.e111 = v_lookup_obj.e111
    self.h100 = v_lookup_obj.h100
    self.h110 = v_lookup_obj.h110
    self.h111 = v_lookup_obj.h111
    self.ea = v_lookup_obj.ea
    self.eb = v_lookup_obj.eb
    self.ec = v_lookup_obj.ec
    self.ebp = v_lookup_obj.ebp
    self.ecp = v_lookup_obj.ecp
    self.ha = v_lookup_obj.ha
    self.hb = v_lookup_obj.hb
    self.hc = v_lookup_obj.hc
    self.hbp = v_lookup_obj.hbp
    self.hcp = v_lookup_obj.hcp
    self.hcorr = v_lookup_obj.hcorr
    self.ecorr = v_lookup_obj.ecorr
  
  def __getstate__(self):
    state = self.__dict__.copy()
    # Remove the unpicklable entries.
    del state['v_lookup_obj']
    return state

  def __setstate__(self, state):
    self.__dict__.update(state)
    self.v_lookup_obj = GATSiggenVelocityLookup()
    
    self.v_lookup_obj.e = self.e
    self.v_lookup_obj.e100 = self.e100
    self.v_lookup_obj.e110 = self.e110
    self.v_lookup_obj.e111 = self.e111
    self.v_lookup_obj.h100 = self.h100
    self.v_lookup_obj.h110 = self.h110
    self.v_lookup_obj.h111 = self.h111
    self.v_lookup_obj.ea = self.ea
    self.v_lookup_obj.eb = self.eb
    self.v_lookup_obj.ec = self.ec
    self.v_lookup_obj.ebp = self.ebp
    self.v_lookup_obj.ecp = self.ecp
    self.v_lookup_obj.ha = self.ha
    self.v_lookup_obj.hb = self.hb
    self.v_lookup_obj.hc = self.hc
    self.v_lookup_obj.hbp = self.hbp
    self.v_lookup_obj.hcp = self.hcp
    self.v_lookup_obj.hcorr = self.hcorr
    self.v_lookup_obj.ecorr = self.ecorr


