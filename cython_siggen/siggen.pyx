# file: cqueue.pxd

from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

from libc.stdlib cimport malloc, free
from libc.string cimport strcpy

import numpy as np
import cython
cimport numpy as np
cimport csiggen

cdef extern from "siggen_helpers.c":
    int read_velocity_table(csiggen.velocity_lookup* v_lookup, csiggen.MJD_Siggen_Setup *setup)
    int temperature_modify_velocity_table( csiggen.velocity_lookup* v_lookup_saved,  csiggen.velocity_lookup* modified_v_lookup, csiggen.MJD_Siggen_Setup *setup)

cdef class Siggen:

  cdef csiggen.MJD_Siggen_Setup fSiggenData
  cdef csiggen.velocity_lookup* fVelocityFileData #as read straight out of the drift velo file
  cdef csiggen.velocity_lookup* fVelocityTempData #temperature-adjuisted values for use in siggen
  
  cdef csiggen.cyl_pt** pEfld;
  cdef float** pWpot;
  
#  cdef csiggen.point* pDpath_e
#  cdef csiggen.point* pDpath_h

  def __init__(self, conffilename="", timeStepLength=-1., numTimeSteps=-1, savedConfig=None):
    
    if savedConfig is not None:
      self.SetConfiguration(savedConfig)
      self.reinit_from_saved_state()
    else:
      csiggen.read_config(conffilename, &self.fSiggenData);
      csiggen.field_setup(&self.fSiggenData);

      if timeStepLength ==-1. or numTimeSteps ==-1:
        self.fSiggenData.ntsteps_out = self.fSiggenData.time_steps_calc / np.int(self.fSiggenData.step_time_out/self.fSiggenData.step_time_calc);
      else:
        self.set_time_step_length(timeStepLength)
        self.set_calc_time_step_length(timeStepLength)
        self.set_time_step_number(numTimeSteps)
    
    self.fSiggenData.dpath_e = <csiggen.point *> PyMem_Malloc(self.fSiggenData.time_steps_calc*sizeof(csiggen.point));
    self.fSiggenData.dpath_h = <csiggen.point *> PyMem_Malloc(self.fSiggenData.time_steps_calc*sizeof(csiggen.point));

    self.ReadVelocityTable()
    self.SetTemperature(self.fSiggenData.xtal_temp)
    
  def __dealloc__(self):
    if self.fSiggenData.dpath_e is not NULL:
      PyMem_Free(self.fSiggenData.dpath_e)
    if self.fSiggenData.dpath_h is not NULL:
      PyMem_Free(self.fSiggenData.dpath_h)
    if self.fVelocityFileData is not NULL:
      PyMem_Free(self.fVelocityFileData)
    if self.fVelocityTempData is not NULL:
      PyMem_Free(self.fVelocityTempData)
    if self.pWpot is not NULL:
      PyMem_Free(self.pWpot)
    if self.pEfld is not NULL:
      PyMem_Free(self.pEfld)
    
  cdef reinit_from_saved_state(self):
    #init the WP and E-fld
    cdef csiggen.cyl_pt **efld;
    efld = <csiggen.cyl_pt **> PyMem_Malloc(self.fSiggenData.rlen*sizeof(csiggen.cyl_pt*));
    for i in range(self.fSiggenData.rlen):
      efld[i] = <csiggen.cyl_pt *> PyMem_Malloc(self.fSiggenData.zlen*sizeof(csiggen.cyl_pt));
    self.fSiggenData.efld = efld;
    
    cdef float** wpot;
    wpot = <float **> malloc(self.fSiggenData.rlen*sizeof(float*));
    for i in range(self.fSiggenData.rlen):
      wpot[i] = <float *> malloc(self.fSiggenData.zlen*sizeof(float));
    self.fSiggenData.wpot = wpot;

    self.pEfld = efld;
    self.pWpot = wpot;
  
  cdef c_get_signal(self, float x, float y, float z, float* signal):
    cdef csiggen.point pt
    pt.x = x
    pt.y = y
    pt.z = z
  
    return csiggen.get_signal( pt, signal, &self.fSiggenData)

  def GetSignal(self, float x, float y, float z, np.ndarray[float, ndim=1, mode="c"] input not None):
    return self.c_get_signal(x,y,z, &input[0])

  cpdef set_time_step_length(self, float timeStepLength):
    if timeStepLength < self.fSiggenData.step_time_calc:
      print "Also reducing time step calc to %f" % timeStepLength
      self.fSiggenData.step_time_calc = timeStepLength;
    self.fSiggenData.step_time_out = timeStepLength

  cpdef set_calc_time_step_length(self, float timeStepOutLength):
      self.fSiggenData.step_time_calc = timeStepOutLength;

  cpdef set_time_step_number(self, int waveformLength):
    self.fSiggenData.ntsteps_out = waveformLength;
    self.fSiggenData.time_steps_calc = waveformLength;

  cdef c_read_velocity_table(self):
    #read in the drift velocity table
    if self.fVelocityFileData is NULL:
      self.fVelocityFileData = <csiggen.velocity_lookup* > malloc(21*sizeof(csiggen.velocity_lookup))
    read_velocity_table(self.fVelocityFileData, &self.fSiggenData)

    #and copy the current version of it to an unsaved array w/ a hard copy
    if self.fVelocityTempData is NULL:
#      print "about to malloc the velocity adjustment with length %d" % self.fSiggenData.v_lookup_len
      self.fVelocityTempData = <csiggen.velocity_lookup* > PyMem_Malloc(self.fSiggenData.v_lookup_len*sizeof(csiggen.velocity_lookup))
#    else:
#      print "about to realloc the velocity adjustment with length %d" % self.fSiggenData.v_lookup_len
#      self.fVelocityTempData = <csiggen.velocity_lookup* > PyMem_Realloc(self.fVelocityTempData, self.fSiggenData.v_lookup_len*sizeof(self.fVelocityTempData))

#    print "about to copy the velocity adjustment table with length %d" % self.fSiggenData.v_lookup_len
    for i in range(self.fSiggenData.v_lookup_len):
#      print "copying index %d..." % i
#      print self.fVelocityFileData[i].e

      self.fVelocityTempData[i].e =     self.fVelocityFileData[i].e
      self.fVelocityTempData[i].e100 =  self.fVelocityFileData[i].e100
      self.fVelocityTempData[i].e110 =  self.fVelocityFileData[i].e110
      self.fVelocityTempData[i].e111 =  self.fVelocityFileData[i].e111
      self.fVelocityTempData[i].h100 =  self.fVelocityFileData[i].h100
      self.fVelocityTempData[i].h110 =  self.fVelocityFileData[i].h110
      self.fVelocityTempData[i].h111 =  self.fVelocityFileData[i].h111
      self.fVelocityTempData[i].ea =    self.fVelocityFileData[i].ea
      self.fVelocityTempData[i].eb =    self.fVelocityFileData[i].eb
      self.fVelocityTempData[i].ec =    self.fVelocityFileData[i].ec
      self.fVelocityTempData[i].ebp =   self.fVelocityFileData[i].ebp
      self.fVelocityTempData[i].ecp =   self.fVelocityFileData[i].ecp
      self.fVelocityTempData[i].ha =    self.fVelocityFileData[i].ha
      self.fVelocityTempData[i].hb =    self.fVelocityFileData[i].hb
      self.fVelocityTempData[i].hc =    self.fVelocityFileData[i].hc
      self.fVelocityTempData[i].hbp =   self.fVelocityFileData[i].hbp
      self.fVelocityTempData[i].hcp =   self.fVelocityFileData[i].hcp
      self.fVelocityTempData[i].hcorr = self.fVelocityFileData[i].hcorr
      self.fVelocityTempData[i].ecorr = self.fVelocityFileData[i].ecorr

  cpdef ReadVelocityTable(self):
    self.c_read_velocity_table()

  def ReadCorrectedVelocity(self):
    for i in range(self.fSiggenData.v_lookup_len):
    
      if i >3: continue
    
      print "Row number %d" % i
      print self.fVelocityTempData[i].e
      print self.fVelocityTempData[i].e100
      print self.fVelocityTempData[i].e110
      print self.fVelocityTempData[i].e111
      print self.fVelocityTempData[i].h100
      print self.fVelocityTempData[i].h110
      print self.fVelocityTempData[i].h111
      print self.fVelocityTempData[i].ea
      print self.fVelocityTempData[i].eb
      print self.fVelocityTempData[i].ec
      print self.fVelocityTempData[i].ebp
      print self.fVelocityTempData[i].ecp
      print self.fVelocityTempData[i].ha
      print self.fVelocityTempData[i].hb
      print self.fVelocityTempData[i].hc
      print self.fVelocityTempData[i].hbp
      print self.fVelocityTempData[i].hcp
      print self.fVelocityTempData[i].hcorr
      print self.fVelocityTempData[i].ecorr

  cpdef SetTemperature(self, float temp):
    self.fSiggenData.xtal_temp = temp
    temperature_modify_velocity_table(self.fVelocityFileData, self.fVelocityTempData, &self.fSiggenData)
    self.fSiggenData.v_lookup = self.fVelocityTempData

  def SetPointContact(self, pcRad, pcLen):
    self.fSiggenData.pc_radius = pcRad
    self.fSiggenData.pc_length = pcLen

  @cython.boundscheck(False)
  @cython.wraparound(False)
  def SetFields(self, new_ef_r, new_ef_z, new_wp):
    for  (i) in range(self.fSiggenData.rlen):
      for (j) in range(self.fSiggenData.zlen):
        self.fSiggenData.efld[i][j].r = new_ef_r[i,j]
        self.fSiggenData.efld[i][j].phi = 0.;
        self.fSiggenData.efld[i][j].z = new_ef_z[i,j]
        self.fSiggenData.wpot[i][j] = new_wp[i,j]

  @cython.boundscheck(False)
  @cython.wraparound(False)
  def ReadFields(self):
    efld_r = np.empty((self.fSiggenData.rlen, self.fSiggenData.zlen))
    efld_phi = np.empty((self.fSiggenData.rlen, self.fSiggenData.zlen))
    efld_z = np.empty((self.fSiggenData.rlen, self.fSiggenData.zlen))
    wp = np.empty((self.fSiggenData.rlen, self.fSiggenData.zlen))
    for  (i) in range(self.fSiggenData.rlen):
      for (j) in range(self.fSiggenData.zlen):
         if self.fSiggenData.efld[i][j].phi != 0:
          print "oh man this is a bummer"
         efld_r[i,j] = self.fSiggenData.efld[i][j].r
         efld_phi[i,j] = self.fSiggenData.efld[i][j].phi
         efld_z[i,j] = self.fSiggenData.efld[i][j].z
         wp[i,j] = self.fSiggenData.wpot[i][j]
    return (efld_r, efld_phi, efld_z, wp)

  def GetTemperature(self):
    return self.fSiggenData.xtal_temp
  def GetDimensions(self):
    return ( self.fSiggenData.xtal_radius, self.fSiggenData.xtal_length)
  
  def FindDriftVelocity(self, float x, float y, float z):
    self.c_find_drift_velocity( x, y, z)

  
  cdef c_find_drift_velocity(self, float x, float y, float z):
    cdef csiggen.point pt
    pt.x = x
    pt.y = y
    pt.z = z
    
    cdef csiggen.vector v
    csiggen.drift_velocity( pt, -1., &v, &self.fSiggenData)
    print "x: %f" % v.x
    print "y: %f" % v.y
    print "z: %f" % v.z


#  @cython.boundscheck(False)
#  @cython.wraparound(False)
#  def SetElectricField(self, np.ndarray[float, ndim=2, mode="c"] efldR not None, np.ndarray[float, ndim=2, mode="c"] efldZ not None):
#    for  i in range(self.fSiggenData.rlen):
#      for j in range(self.fSiggenData.zlen):
#        self.fSiggenData.efld[i][j].r = efldR[i][j];
#        self.fSiggenData.efld[i][j].phi = 0.;
#        self.fSiggenData.efld[i][j].z = efldZ[i][j];
#
#  def SetWeightingPotential(self, np.ndarray[float, ndim=2, mode="c"] newWpot not None):
#    self.c_set_weighting_potential( &newWpot[0] )
#
#  cdef c_set_weighting_potential(self, float** newWpot):
#  #Try to set without copying to make it speedy
#  #NB: May be a terrible idea.  The "Free"-ing will DEFINITELY go poorly.
#  #Live fast, die young YOLO
#    self.fSiggenData.wpot = newWpot;


  def GetSafeConfiguration(self):
  
    siggenConfig = {}
    siggenConfig["verbosity"]  = self.fSiggenData.verbosity;              # 0 = terse, 1 = normal, 2 = chatty/verbose

    # geometry
    siggenConfig["xtal_length"]  = self.fSiggenData.xtal_length;          # z length
    siggenConfig["xtal_radius"]  = self.fSiggenData.xtal_radius;          # radius
    siggenConfig["top_bullet_radius"]  = self.fSiggenData.top_bullet_radius;    # bulletization radius at top of crystal
    siggenConfig["bottom_bullet_radius"]  = self.fSiggenData.bottom_bullet_radius; # bulletization radius at bottom of BEGe crystal
    siggenConfig["pc_length"]  = self.fSiggenData.pc_length;            # point contact length
    siggenConfig["pc_radius"]  = self.fSiggenData.pc_radius;            # point contact radius
    siggenConfig["taper_length"]  = self.fSiggenData.taper_length;         # size of 45-degree taper at bottom of ORTEC-type crystal
    siggenConfig["wrap_around_radius"]  = self.fSiggenData.wrap_around_radius;   # wrap-around radius for BEGes. Set to zero for ORTEC
    siggenConfig["ditch_depth"]  = self.fSiggenData.ditch_depth;          # depth of ditch next to wrap-around for BEGes. Set to zero for ORTEC
    siggenConfig["ditch_thickness"]  = self.fSiggenData.ditch_thickness;      # width of ditch next to wrap-around for BEGes. Set to zero for ORTEC
    siggenConfig["Li_thickness"]  = self.fSiggenData.Li_thickness;         # depth of full-charge-collection boundary for Li contact

    # electric fields & weighing potentials
    siggenConfig["xtal_grid"]  = self.fSiggenData.xtal_grid;            # grid size in mm for field files (either 0.5 or 0.1 mm)
    siggenConfig["impurity_z0"]  = self.fSiggenData.impurity_z0;          # net impurity concentration at Z=0, in 1e10 e/cm3
    siggenConfig["impurity_gradient"]  = self.fSiggenData.impurity_gradient;    # net impurity gradient, in 1e10 e/cm4
    siggenConfig["impurity_quadratic"]  = self.fSiggenData.impurity_quadratic;   # net impurity difference from linear, at z=L/2, in 1e10 e/cm3
    siggenConfig["impurity_surface"]  = self.fSiggenData.impurity_surface;     # surface impurity of passivation layer, in 1e10 e/cm2
    siggenConfig["impurity_radial_add"]  = self.fSiggenData.impurity_radial_add;  # additive radial impurity at outside radius, in 1e10 e/cm3
    siggenConfig["impurity_radial_mult"]  = self.fSiggenData.impurity_radial_mult; # multiplicative radial impurity at outside radius (neutral=1.0)
    siggenConfig["impurity_rpower"]  = self.fSiggenData.impurity_rpower;      # power for radial impurity increase with radius
    siggenConfig["xtal_HV"]  = self.fSiggenData.xtal_HV;              # detector bias for fieldgen, in Volts
    siggenConfig["max_iterations"]  = self.fSiggenData.max_iterations ;       # maximum number of iterations to use in mjd_fieldgen
    siggenConfig["write_field"]  = self.fSiggenData.write_field;          # set to 1 to write V and E to output file, 0 otherwise
    siggenConfig["write_WP"]  = self.fSiggenData.write_WP;             # set to 1 to calculate WP and write it to output file, 0 otherwise
    siggenConfig["bulletize_PC"]  = self.fSiggenData.bulletize_PC;         # set to 1 for inside of point contact hemispherical, 0 for cylindrical

    siggenConfig["drift_name"] = str(self.fSiggenData.drift_name);

    # signal calculation 
    siggenConfig["xtal_temp"]  = self.fSiggenData.xtal_temp;            # crystal temperature in Kelvin
    siggenConfig["preamp_tau"]  = self.fSiggenData.preamp_tau;           # integration time constant for preamplifier, in ns
    siggenConfig["time_steps_calc"]  = self.fSiggenData.time_steps_calc;      # number of time steps used in calculations
    siggenConfig["step_time_calc"]  = self.fSiggenData.step_time_calc;       # length of time step used for calculation, in ns
    siggenConfig["step_time_out"]  = self.fSiggenData.step_time_out;        # length of time step for output signal, in ns
    #    nonzero values in the next few lines significantly slow down the code
    siggenConfig["charge_cloud_size"]  = self.fSiggenData.charge_cloud_size;    # initial FWHM of charge cloud, in mm; set to zero for point charges
    siggenConfig["use_diffusion"]  = self.fSiggenData.use_diffusion;        # set to 0/1 for ignore/add diffusion as the charges drift
    siggenConfig["energy"]  = self.fSiggenData.energy;               # set to energy > 0 to use charge cloud self-repulsion, in keV

    siggenConfig["coord_type"]  = self.fSiggenData.coord_type;           # set to CART or CYL for input point coordinate system
    siggenConfig["ntsteps_out"]  = self.fSiggenData.ntsteps_out;          # number of time steps in output signal

    # data for fields.c
    siggenConfig["rmin"]  = self.fSiggenData.rmin;
    siggenConfig["rmax"]  = self.fSiggenData.rmax;
    siggenConfig["rstep"]  = self.fSiggenData.rstep;
    siggenConfig["zmin"]  = self.fSiggenData.zmin;
    siggenConfig["zmax"]  = self.fSiggenData.zmax;
    siggenConfig["zstep"]  = self.fSiggenData.zstep;
    
    siggenConfig["rlen"]  = self.fSiggenData.rlen;
    siggenConfig["zlen"]  = self.fSiggenData.zlen;           # dimensions of efld and wpot arrays
    siggenConfig["v_lookup_len"]  = self.fSiggenData.v_lookup_len;
    
    # data for calc_signal.c
    siggenConfig["initial_vel"]  = self.fSiggenData.initial_vel;
    siggenConfig["final_vel"]  = self.fSiggenData.final_vel;  # initial and final drift velocities for charges collected to PC
    siggenConfig["dv_dE"]  = self.fSiggenData.dv_dE;     # derivative of drift velocity with field ((mm/ns) / (V/cm))
    siggenConfig["v_over_E"]  = self.fSiggenData.v_over_E;  # ratio of drift velocity to field ((mm/ns) / (V/cm))
    siggenConfig["final_charge_size"]  = self.fSiggenData.final_charge_size;     # in mm
    
    return siggenConfig;

  def SetConfiguration(self, siggenConfig):
  
    self.fSiggenData.verbosity = siggenConfig["verbosity"];              # 0 = terse, 1 = normal, 2 = chatty/verbose

    # geometry
    self.fSiggenData.xtal_length = siggenConfig["xtal_length"];          # z length
    self.fSiggenData.xtal_radius = siggenConfig["xtal_radius"];          # radius
    self.fSiggenData.top_bullet_radius = siggenConfig["top_bullet_radius"];    # bulletization radius at top of crystal
    self.fSiggenData.bottom_bullet_radius = siggenConfig["bottom_bullet_radius"]; # bulletization radius at bottom of BEGe crystal
    self.fSiggenData.pc_length = siggenConfig["pc_length"];            # point contact length
    self.fSiggenData.pc_radius = siggenConfig["pc_radius"];            # point contact radius
    self.fSiggenData.taper_length = siggenConfig["taper_length"];         # size of 45-degree taper at bottom of ORTEC-type crystal
    self.fSiggenData.wrap_around_radius = siggenConfig["wrap_around_radius"];   # wrap-around radius for BEGes. Set to zero for ORTEC
    self.fSiggenData.ditch_depth = siggenConfig["ditch_depth"];          # depth of ditch next to wrap-around for BEGes. Set to zero for ORTEC
    self.fSiggenData.ditch_thickness = siggenConfig["ditch_thickness"];      # width of ditch next to wrap-around for BEGes. Set to zero for ORTEC
    self.fSiggenData.Li_thickness = siggenConfig["Li_thickness"];         # depth of full-charge-collection boundary for Li contact

    # electric fields & weighing potentials
    self.fSiggenData.xtal_grid = siggenConfig["xtal_grid"];            # grid size in mm for field files (either 0.5 or 0.1 mm)
    self.fSiggenData.impurity_z0 = siggenConfig["impurity_z0"];          # net impurity concentration at Z=0, in 1e10 e/cm3
    self.fSiggenData.impurity_gradient = siggenConfig["impurity_gradient"];    # net impurity gradient, in 1e10 e/cm4
    self.fSiggenData.impurity_quadratic = siggenConfig["impurity_quadratic"];   # net impurity difference from linear, at z=L/2, in 1e10 e/cm3
    self.fSiggenData.impurity_surface = siggenConfig["impurity_surface"];     # surface impurity of passivation layer, in 1e10 e/cm2
    self.fSiggenData.impurity_radial_add = siggenConfig["impurity_radial_add"];  # additive radial impurity at outside radius, in 1e10 e/cm3
    self.fSiggenData.impurity_radial_mult = siggenConfig["impurity_radial_mult"]; # multiplicative radial impurity at outside radius (neutral=1.0)
    self.fSiggenData.impurity_rpower = siggenConfig["impurity_rpower"];      # power for radial impurity increase with radius
    self.fSiggenData.xtal_HV = siggenConfig["xtal_HV"];              # detector bias for fieldgen, in Volts
    self.fSiggenData.max_iterations = siggenConfig["max_iterations"];       # maximum number of iterations to use in mjd_fieldgen
    self.fSiggenData.write_field = siggenConfig["write_field"];          # set to 1 to write V and E to output file, 0 otherwise
    self.fSiggenData.write_WP = siggenConfig["write_WP"];             # set to 1 to calculate WP and write it to output file, 0 otherwise
    self.fSiggenData.bulletize_PC = siggenConfig["bulletize_PC"];         # set to 1 for inside of point contact hemispherical, 0 for cylindrical

    strcpy(self.fSiggenData.drift_name,  siggenConfig["drift_name"]);

    # signal calculation 
    self.fSiggenData.xtal_temp = siggenConfig["xtal_temp"];            # crystal temperature in Kelvin
    self.fSiggenData.preamp_tau = siggenConfig["preamp_tau"];           # integration time constant for preamplifier, in ns
    self.fSiggenData.time_steps_calc = siggenConfig["time_steps_calc"];      # number of time steps used in calculations
    self.fSiggenData.step_time_calc = siggenConfig["step_time_calc"];       # length of time step used for calculation, in ns
    self.fSiggenData.step_time_out = siggenConfig["step_time_out"];        # length of time step for output signal, in ns
    #    nonzero values in the next few lines significantly slow down the code
    self.fSiggenData.charge_cloud_size = siggenConfig["charge_cloud_size"];    # initial FWHM of charge cloud, in mm"]; set to zero for point charges
    self.fSiggenData.use_diffusion = siggenConfig["use_diffusion"];        # set to 0/1 for ignore/add diffusion as the charges drift
    self.fSiggenData.energy = siggenConfig["energy"];               # set to energy > 0 to use charge cloud self-repulsion, in keV

    self.fSiggenData.coord_type = siggenConfig["coord_type"];           # set to CART or CYL for input point coordinate system
    self.fSiggenData.ntsteps_out = siggenConfig["ntsteps_out"];          # number of time steps in output signal

    # data for fields.c
    self.fSiggenData.rmin = siggenConfig["rmin"];
    self.fSiggenData.rmax = siggenConfig["rmax"];
    self.fSiggenData.rstep = siggenConfig["rstep"];
    self.fSiggenData.zmin = siggenConfig["zmin"];
    self.fSiggenData.zmax = siggenConfig["zmax"];
    self.fSiggenData.zstep = siggenConfig["zstep"];
    
    self.fSiggenData.rlen = siggenConfig["rlen"];
    self.fSiggenData.zlen = siggenConfig["zlen"];           # dimensions of efld and wpot arrays
    self.fSiggenData.v_lookup_len = siggenConfig["v_lookup_len"];
    
    # data for calc_signal.c
    self.fSiggenData.initial_vel = siggenConfig["initial_vel"];
    self.fSiggenData.final_vel = siggenConfig["final_vel"];  # initial and final drift velocities for charges collected to PC
    self.fSiggenData.dv_dE = siggenConfig["dv_dE"];     # derivative of drift velocity with field ((mm/ns) / (V/cm))
    self.fSiggenData.v_over_E = siggenConfig["v_over_E"];  # ratio of drift velocity to field ((mm/ns) / (V/cm))
    self.fSiggenData.final_charge_size = siggenConfig["final_charge_size"];     # in mm
