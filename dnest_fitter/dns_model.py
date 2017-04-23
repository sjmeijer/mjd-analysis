import numpy as np
import scipy.stats as stats
import dnest4

import numpy.random as rng

#some values for priors

traprc_min = 150
tf_phi_max = -1.2

tf_first_idx = 0
velo_first_idx = 7

grad_idx = velo_first_idx + 6
imp_avg_idx = grad_idx + 1
trap_idx = imp_avg_idx + 1

ba_idx, c_idx, dc_idx = np.arange(3)+ tf_first_idx
rc1_idx, rc2_idx, rcfrac_idx = np.arange(3)+ tf_first_idx+3
aliasrc_idx = tf_first_idx+6

ba_prior = 0.107213
c_prior = -0.808
dc_prior = 0.815/c_prior

rc1_prior =  73.085166
rc2_prior = 1.138420
rc_frac_prior = 0.997114

# h_100_mu0_prior, h_100_beta_prior, h_100_e0_prior = 66333., 0.744, 181.
# h_111_mu0_prior, h_111_beta_prior, h_111_e0_prior =  107270., 0.580, 100.

E_a = 500
va_lims = [2.5E6, 10E6]
vmax_lims = [7.5E6, 1.5E7]
beta_lims = [0.4, 1]

priors = np.empty(trap_idx+1) #6 + 2)
prior_vars =  np.empty(len(priors))

priors[rc1_idx:rc1_idx+3] = rc1_prior, rc2_prior, rc_frac_prior
prior_vars[rc1_idx:rc1_idx+3] = 0.2, 0.3, 0.001

priors[aliasrc_idx] = 3.
prior_vars[aliasrc_idx] = 3.

priors[grad_idx] = 100
prior_vars[grad_idx] = 3
priors[trap_idx] = 200.
prior_vars[trap_idx] = 25.

def global_get_indices():
    return (tf_first_idx, velo_first_idx, grad_idx, trap_idx)

class Model(object):
    """
    Specify the model in Python.
    """
    def __init__(self, mpi_manager):
        """
        Parameter values *are not* stored inside the class
        """
        self.mpi_manager = mpi_manager
        self.maxt_guess = self.mpi_manager.max_sample_idx
        self.max_maxt = self.maxt_guess + 10
        self.min_maxt = self.maxt_guess - 10

        num_waveforms = self.mpi_manager.num_waveforms
        self.changed_wfs = np.zeros(num_waveforms)
        self.ln_likes = np.zeros(num_waveforms)

    def draw_position(self, wf_idx):
      r = rng.rand() * self.mpi_manager.detector.detector_radius
      z = rng.rand() * self.mpi_manager.detector.detector_length
      scale = np.amax(self.mpi_manager.wfs[wf_idx].windowedWf)
      t0 = None

      if not self.mpi_manager.detector.IsInDetector(r, 0.1, z):
        return self.draw_position(wf_idx)
      else:
        return (r,z, scale, t0)

    def from_prior(self):
        """
        Unlike in C++, this must *return* a numpy array of parameters.
        """
        num_waveforms = self.mpi_manager.num_waveforms

        r_arr      = np.empty(num_waveforms)
        z_arr      = np.empty(num_waveforms)

        rad_arr      = np.empty(num_waveforms)
        phi_arr    = np.empty(num_waveforms)
        theta_arr  = np.empty(num_waveforms)
        scale_arr  = np.empty(num_waveforms)
        t0_arr     = np.empty(num_waveforms)
        smooth_arr = np.empty(num_waveforms)
        m_arr      = np.empty(num_waveforms)
        b_arr      = np.empty(num_waveforms)

        #draw 8 waveform params for each waveform
        for (wf_idx) in range(num_waveforms):
            (r,z, scale, t0) = self.draw_position(wf_idx)
            rad = np.sqrt(r**2+z**2)
            theta = np.arctan(z/r)

            smooth_guess = 10

            r_arr[wf_idx] = r
            z_arr[wf_idx] = z
            rad_arr[wf_idx] = rad
            phi_arr[wf_idx] = rng.rand() * np.pi/4
            theta_arr[wf_idx] = theta
            scale_arr[wf_idx] = 5*rng.randn() + scale - .005*scale
            t0_arr[wf_idx] = 3*rng.randn() + self.maxt_guess
            smooth_arr[wf_idx] = np.clip(rng.randn() + smooth_guess, 0, 20)
            # m_arr[wf_idx] =  0.0001*rng.randn() + 0.
            # b_arr[wf_idx] =  0.001*rng.randn() + 0.

        phi = (tf_phi_max + np.pi/2) * rng.rand() - np.pi/2
        omega = np.pi * rng.rand()
        d = rng.rand()

        #limit from 60 to 90
        rc1 = dnest4.wrap(prior_vars[rc1_idx]*rng.randn() + priors[rc1_idx], 65, 80)
        rc2 = dnest4.wrap(prior_vars[rc2_idx]*rng.randn() + priors[rc2_idx], 0.1, 10)
        rcfrac = dnest4.wrap(prior_vars[rcfrac_idx]*rng.randn() + priors[rcfrac_idx], 0.9, 1)

        grad = rng.rand()*( self.mpi_manager.detector.gradList[-1] - self.mpi_manager.detector.gradList[0] ) + self.mpi_manager.detector.gradList[0]
        avgImp = rng.rand()*( self.mpi_manager.detector.impAvgList[-1] - self.mpi_manager.detector.impAvgList[0] ) + self.mpi_manager.detector.impAvgList[0]
        charge_trapping = rng.rand()*(1000 - traprc_min) +  traprc_min

        aliasrc = dnest4.wrap(prior_vars[aliasrc_idx]*rng.randn() + priors[aliasrc_idx], 0.01, 10)

        #6 hole drift params
        h_100_va = (va_lims[1] - va_lims[0]) * rng.rand() + va_lims[0]
        h_100_vmax = (vmax_lims[1] - vmax_lims[0]) * rng.rand() + vmax_lims[0]
        h_100_beta = (beta_lims[1] - beta_lims[0]) * rng.rand() + beta_lims[0]

        h_111_va = (va_lims[1] - va_lims[0]) * rng.rand() + va_lims[0]
        h_111_vmax = (vmax_lims[1] - vmax_lims[0]) * rng.rand() + vmax_lims[0]
        h_111_beta = (beta_lims[1] - beta_lims[0]) * rng.rand() + beta_lims[0]


        return np.hstack([
              phi, omega, d,
              #b, c, dc,
              rc1, rc2, rcfrac,aliasrc,
              h_100_va, h_111_va, h_100_vmax, h_111_vmax, h_100_beta, h_111_beta,
              grad, avgImp, charge_trapping,
              rad_arr[:], phi_arr[:], theta_arr[:], scale_arr[:], t0_arr[:],smooth_arr[:],
            ])

    def perturb(self, params):
        """
        Unlike in C++, this takes a numpy array of parameters as input,
        and modifies it in-place. The return value is still logH.
        """
        logH = 0.0
        num_waveforms = self.mpi_manager.num_waveforms

        #decide whether to adjust just waveforms, or both wfs and detector params

        if rng.rand() <= 0.5:
            #adjust detector only
            reps = 1;
            if(rng.rand() < 0.5):
                reps += np.int(np.power(100.0, rng.rand()));

            for i in range(reps):
                which = rng.randint(len(priors))
                logH += self.perturb_detector(params, which)
        else:
            #adjust at least one waveform:
            self.changed_wfs.fill(0)
            randt2 = rng.randn()/np.sqrt(-np.log(rng.rand()));
            chance = np.power(10.0, -3*np.abs(randt2));

            for wf_idx in range(num_waveforms):
                if rng.rand() <= chance:
                     self.changed_wfs[wf_idx] = True
                #make sure one waveform is changed:
            if np.any(self.changed_wfs) == 0:
                self.changed_wfs[rng.randint(num_waveforms)] = 1

            for wf_idx in range(num_waveforms):
                if self.changed_wfs[wf_idx] == 1:
                    logH += self.perturb_wf(params, wf_idx)
        return logH

    def perturb_detector(self, params, which):
        logH = 0.0
        detector = self.mpi_manager.detector
        num_waveforms =  self.mpi_manager.num_waveforms

        if which == ba_idx: #lets call this phi
           params[which] += (tf_phi_max + np.pi/2)*dnest4.randh()
           params[which] = dnest4.wrap(params[which], -np.pi/2, tf_phi_max)
        elif which == c_idx: #call it omega
            params[which] += 0.1*dnest4.randh()
            params[which] = dnest4.wrap(params[which], 0.07, 0.2)
        elif which == dc_idx: #d
            params[which] += 0.01*dnest4.randh()
            params[which] = dnest4.wrap(params[which], 0.7, 0.9)

        elif which == rc1_idx or which == rc2_idx or which == rcfrac_idx:
            #all normally distributed priors
            logH -= -0.5*((params[which] - priors[which])/prior_vars[which])**2
            params[which] += prior_vars[which]*dnest4.randh()
            logH += -0.5*((params[which] - priors[which])/prior_vars[which])**2

        elif which == aliasrc_idx:
          params[which] += 19.9*dnest4.randh()
          params[which] = dnest4.wrap(params[which], 0.1, 20)

        elif which == grad_idx:
          params[which] += (detector.gradList[-1] - detector.gradList[0])*dnest4.randh()
          params[which] = dnest4.wrap(params[which], detector.gradList[0], detector.gradList[-1])
        elif which == imp_avg_idx:
          params[which] += (detector.impAvgList[-1] - detector.impAvgList[0])*dnest4.randh()
          params[which] = dnest4.wrap(params[which], detector.impAvgList[0], detector.impAvgList[-1])
        elif which == trap_idx:
            params[which] += (1000 - traprc_min)*dnest4.randh()
            params[which] = dnest4.wrap(params[which], traprc_min, 1000)

        elif which == velo_first_idx:
          params[which] += (va_lims[1] - va_lims[0])  *dnest4.randh()
          params[which] = dnest4.wrap(params[which], va_lims[0], va_lims[1])
        elif which == velo_first_idx+1:
          params[which] += (vmax_lims[1] - vmax_lims[0])  *dnest4.randh()
          params[which] = dnest4.wrap(params[which], vmax_lims[0], vmax_lims[1])
        elif which == velo_first_idx+2 or  which == velo_first_idx+3:
          params[which] += 1.  *dnest4.randh()
          params[which] = dnest4.wrap(params[which], 1., 2.)
        elif which == velo_first_idx+4 or which == velo_first_idx+5:
          params[which] += (beta_lims[1] - beta_lims[0])  *dnest4.randh()
          params[which] = dnest4.wrap(params[which], beta_lims[0], beta_lims[1])

        else: #velocity or rc params: cant be below 0, can be arb. large
            print ("which value %d not supported" % which)
            exit(0)

        return logH


    def perturb_wf(self, params, wf_idx, ):
    #do both wf and detector params in case theres strong correlation
        logH = 0.0
        num_waveforms = self.mpi_manager.num_waveforms
        detector = self.mpi_manager.detector

        reps = 1
        if rng.rand() < 0.5:
            reps += np.int(np.power(100.0, rng.rand()));

        for i in range(reps):
            wf_which = rng.randint(6)

            # my_which = rng.randint(len(priors) + 8)

            # if my_which < len(priors):
            #     #detector variable
            #     logH += self.perturb_detector(params, my_which)
            #
            # else:
            if wf_which < 6:
                #this is a waveform variable!
                # wf_which =  np.int(my_which - len(priors))

                #which idx of the global params array
                which = len(priors) + wf_which*num_waveforms + wf_idx

                rad_idx = len(priors) + wf_idx
                theta_idx =  len(priors) + 2*num_waveforms+ wf_idx
                self.changed_wfs[wf_idx] = 1

                if wf_which == 0:
                  theta = params[theta_idx]

                  #FIND THE MAXIMUM RADIUS STILL INSIDE THE DETECTOR
                  theta_eq = np.arctan(detector.detector_length/detector.detector_radius)
                  theta_taper = np.arctan(detector.taper_length/detector.detector_radius)
                  if theta <= theta_taper:
                     z = np.tan(theta)*(detector.detector_radius - detector.taper_length) / (1-np.tan(theta))
                     max_rad = z / np.sin(theta)
                  elif theta <= theta_eq:
                      max_rad = detector.detector_radius / np.cos(theta)
                  else:
                      theta_comp = np.pi/2 - theta
                      max_rad = detector.detector_length / np.cos(theta_comp)

                  #AND THE MINIMUM (from PC dimple)
                  #min_rad  = 1./ ( np.cos(theta)**2/detector.pcRad**2  +  np.sin(theta)**2/detector.pcLen**2 )

                  min_rad = np.amax([detector.pcRad, detector.pcLen])

                  total_max_rad = np.sqrt(detector.detector_length**2 + detector.detector_radius**2 )

                  params[which] += total_max_rad*dnest4.randh()
                  params[which] = dnest4.wrap(params[which] , min_rad, max_rad)

                elif wf_which ==2: #theta
                  rad = params[rad_idx]

                  if rad < np.amin([detector.detector_radius - detector.taper_length, detector.detector_length]):
                      max_val = np.pi/2
                      min_val = 0
                  else:
                      if rad < detector.detector_radius - detector.taper_length:
                          #can't possibly hit the taper
                          min_val = 0
                      elif rad < np.sqrt(detector.detector_radius**2 + detector.taper_length**2):
                          #low enough that it could hit the taper region
                          a = detector.detector_radius - detector.taper_length
                          z = 0.5 * (np.sqrt(2*rad**2-a**2) - a)
                          min_val = np.arcsin(z/rad)
                      else:
                          #longer than could hit the taper
                          min_val = np.arccos(detector.detector_radius/rad)

                      if rad < detector.detector_length:
                          max_val = np.pi/2
                      else:
                          max_val = np.pi/2 - np.arccos(detector.detector_length/rad)

                  params[which] += np.pi/2*dnest4.randh()
                  params[which] = dnest4.wrap(params[which], min_val, max_val)

                # if wf_which == 0:
                #     params[which] += (detector.detector_radius)*dnest4.randh()
                #     params[which] = dnest4.wrap(params[which] , 0, detector.detector_radius)
                elif wf_which == 1:
                    max_val = np.pi/4
                    params[which] += np.pi/4*dnest4.randh()
                    params[which] = dnest4.wrap(params[which], 0, max_val)
                    if params[which] < 0 or params[which] > np.pi/4:
                        print ("wtf phi")

                # elif wf_which == 2:
                #     params[which] += (detector.detector_length)*dnest4.randh()
                #     params[which] = dnest4.wrap(params[which] , 0, detector.detector_length)

                elif wf_which == 3: #scale
                    wf = self.mpi_manager.wfs[wf_idx]
                    min_scale = wf.wfMax - 0.01*wf.wfMax
                    max_scale = wf.wfMax + 0.005*wf.wfMax
                    params[which] += (max_scale-min_scale)*dnest4.randh()
                    params[which] = dnest4.wrap(params[which], min_scale, max_scale)

                elif wf_which == 4: #t0
                  params[which] += 1*dnest4.randh()
                  params[which] = dnest4.wrap(params[which], self.min_maxt, self.max_maxt)
                elif wf_which == 5: #smooth
                  params[which] += 0.1*dnest4.randh()
                  params[which] = dnest4.wrap(params[which], 0, 25)

                else:
                    print( "wf which value %d (which value %d) not supported" % (wf_which, which) )
                    exit(0)

        return logH

    def log_likelihood(self, params):
        return self.mpi_manager.calc_likelihood(params)

    def get_indices(self):
        return (tf_first_idx, velo_first_idx, trap_idx, grad_idx, len(priors))
