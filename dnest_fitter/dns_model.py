import numpy as np
import scipy.stats as stats
import scipy.optimize as op
import dnest4

import numpy.random as rng
from multiprocessing import Pool


traprc_min = 100

tf_first_idx = 0
velo_first_idx = 6
trap_idx = 13
grad_idx = 12

priors = np.empty(6 + 6 + 1 + 1) #6 + 2)

ba_idx, c_idx, dc_idx = np.arange(3)+ tf_first_idx
rc1_idx, rc2_idx, rcfrac_idx = np.arange(3)+ tf_first_idx+3

ba_prior = 0.107213
c_prior = -0.808
dc_prior = 0.815/c_prior

rc1_prior =  73.085166
rc2_prior = 1.138420
rc_frac_prior = 0.997114

h_100_mu0_prior, h_100_beta_prior, h_100_e0_prior = 66333., 0.744, 181.
h_111_mu0_prior, h_111_beta_prior, h_111_e0_prior =  107270., 0.580, 100.

priors[rc1_idx:rc1_idx+3] = rc1_prior, rc2_prior, rc_frac_prior
priors[velo_first_idx:velo_first_idx+3] = h_100_mu0_prior, h_100_beta_prior, h_100_e0_prior
priors[velo_first_idx+3:velo_first_idx+6] = h_111_mu0_prior, h_111_beta_prior, h_111_e0_prior

prior_vars =  np.empty(len(priors))
prior_vars[rc1_idx:rc1_idx+3] = 0.2, 0.3, 0.001

velo_width = 10.
velo_var = 1.
prior_vars[velo_first_idx:velo_first_idx+6] = velo_var*priors[velo_first_idx:velo_first_idx+6]

priors[grad_idx] = 100
prior_vars[grad_idx] = 3
priors[trap_idx] = 200.
prior_vars[trap_idx] = 25.



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

        print "\n"
        #draw 8 waveform params for each waveform
        for (wf_idx) in range(num_waveforms):
            (r,z, scale, t0) = self.draw_position(wf_idx)
            smooth_guess = 10

            r_arr[wf_idx] = r
            z_arr[wf_idx] = z
            # rad_arr[wf_idx] = rad
            phi_arr[wf_idx] = rng.rand() * np.pi/4
            # theta_arr[wf_idx] = theta
            scale_arr[wf_idx] = 5*rng.randn() + scale - .005*scale
            t0_arr[wf_idx] = 3*rng.randn() + self.maxt_guess
            smooth_arr[wf_idx] = np.clip(rng.randn() + smooth_guess, 0, 20)
            m_arr[wf_idx] =  0.0001*rng.randn() + 0.
            b_arr[wf_idx] =  0.001*rng.randn() + 0.

        phi = np.pi * rng.rand() - np.pi/2
        omega = np.pi * rng.rand()
        d = rng.rand()

        #limit from 60 to 90
        rc1 = dnest4.wrap(prior_vars[rc1_idx]*rng.randn() + priors[rc1_idx], 65, 80)
        rc2 = dnest4.wrap(prior_vars[rc2_idx]*rng.randn() + priors[rc2_idx], 0.1, 10)
        rcfrac = dnest4.wrap(prior_vars[rcfrac_idx]*rng.randn() + priors[rcfrac_idx], 0.9, 1)
        charge_trapping = dnest4.wrap(prior_vars[trap_idx]*rng.randn() + priors[trap_idx], 50, 1000)

        grad = np.int(np.clip(prior_vars[grad_idx]*np.int(rng.randn()) + priors[grad_idx], 0, len(self.mpi_manager.detector.gradList)-1))

        #6 hole drift params
        h_100_mu0 = .01 * h_100_mu0_prior*rng.randn() + h_100_mu0_prior
        lnbeta = np.log(1/(.1*h_100_beta_prior))*rng.randn() + np.log(1./h_100_beta_prior)
        h_100_lnbeta = dnest4.wrap(lnbeta, 0, np.log(1/.1))
        h_100_emu = .01 * (h_100_e0_prior*h_100_mu0_prior)*rng.randn() + h_100_e0_prior*h_100_mu0_prior

        h_111_mu0 = .01 * h_111_mu0_prior*rng.randn() + h_111_mu0_prior
        lnbeta = np.log(1/(.1*h_111_beta_prior))*rng.randn() + np.log(1./h_111_e0_prior)
        h_111_lnbeta = dnest4.wrap(lnbeta, 0, np.log(1/.1))
        h_111_emu = .01 * (h_111_e0_prior*h_111_mu0_prior)*rng.randn() + h_111_e0_prior*h_111_mu0_prior


        return np.hstack([
              phi, omega, d,
              #b, c, dc,
              rc1, rc2, rcfrac,
              h_100_mu0, h_100_lnbeta, h_100_emu, h_111_mu0, h_111_lnbeta, h_111_emu,
              grad, charge_trapping,
              r_arr[:], phi_arr[:], z_arr[:], scale_arr[:], t0_arr[:],smooth_arr[:], m_arr[:], b_arr[:]
            ])

    def perturb(self, params):
        """
        Unlike in C++, this takes a numpy array of parameters as input,
        and modifies it in-place. The return value is still logH.
        """
        logH = 0.0
        detector = self.mpi_manager.detector
        num_waveforms = self.mpi_manager.num_waveforms

        det_or_wf = rng.randint(2)
        if det_or_wf == 0:
            which = rng.randint(len(priors))
        else:
            which = rng.randint(num_waveforms*8) + len(priors)

        if which >= len(priors):
            #this is a waveform variable!
            wf_which =  np.int(np.floor((which - len(priors)) / num_waveforms))
            wf_idx = (which - len(priors)) % num_waveforms

            if wf_which == 0:
                params[which] += (detector.detector_radius)*dnest4.randh()
                params[which] = dnest4.wrap(params[which] , 0, detector.detector_radius)
            elif wf_which == 1:
                max_val = np.pi/4
                params[which] += np.pi/4*dnest4.randh()
                params[which] = dnest4.wrap(params[which], 0, max_val)
                if params[which] < 0 or params[which] > np.pi/4:
                    print "wtf phi"
                #params[which] = np.clip(params[which], 0, max_val)
            elif wf_which == 2:
                params[which] += (detector.detector_length)*dnest4.randh()
                params[which] = dnest4.wrap(params[which] , 0, detector.detector_length)

            elif wf_which == 3: #scale
                wf = self.mpi_manager.wfs[wf_idx]
                min_scale = wf.wfMax - 0.01*wf.wfMax
                max_scale = wf.wfMax + 0.005*wf.wfMax
                params[which] += (max_scale-min_scale)*dnest4.randh()
                params[which] = dnest4.wrap(params[which], min_scale, max_scale)
            #   print "  adjusted scale to %f" %  ( params[which])

            elif wf_which == 4: #t0
              params[which] += 1*dnest4.randh()
              params[which] = dnest4.wrap(params[which], self.min_maxt, self.max_maxt)
            elif wf_which == 5: #smooth
              params[which] += 0.1*dnest4.randh()
              params[which] = dnest4.wrap(params[which], 0, 25)
            #   print "  adjusted smooth to %f" %  ( params[which])

            elif wf_which == 6: #wf baseline slope (m)
                logH -= -0.5*(params[which]/1E-4)**2
                params[which] += 1E-4*dnest4.randh()
                logH += -0.5*(params[which]/1E-4)**2
            elif wf_which == 7: #wf baseline incercept (b)
                logH -= -0.5*(params[which]/1E-2)**2
                params[which] += 1E-2*dnest4.randh()
                logH += -0.5*(params[which]/1E-2)**2

            #   params[which] += 0.01*dnest4.randh()
            #   params[which]=dnest4.wrap(params[which], -1, 1)
            #   print "  adjusted b to %f" %  ( params[which])

        elif which == ba_idx: #lets call this phi
           params[which] += np.pi*dnest4.randh()
           params[which] = dnest4.wrap(params[which], -np.pi/2, np.pi/2)
        elif which == c_idx: #call it omega
            params[which] += 0.1*dnest4.randh()
            params[which] = dnest4.wrap(params[which], 0.13, 0.14)
        elif which == dc_idx: #d
            params[which] += 0.01*dnest4.randh()
            params[which] = dnest4.wrap(params[which], 0.81, 0.82)

        elif which == rc1_idx or which == rc2_idx or which == rcfrac_idx:
            #all normally distributed priors
            logH -= -0.5*((params[which] - priors[which])/prior_vars[which])**2
            params[which] += prior_vars[which]*dnest4.randh()
            logH += -0.5*((params[which] - priors[which])/prior_vars[which])**2

        elif which == grad_idx:
          params[which] += (len(detector.gradList)-1)*dnest4.randh()
          params[which] = np.int(dnest4.wrap(params[which], 0, len(detector.gradList)-1))
        elif which >= velo_first_idx and which < velo_first_idx+6:
            mu_max = 100E5
            velo_which =  (which - velo_first_idx)%3
            #TODO: consider long transforming priors on two of these
            if velo_which ==0: #mu0 parameter
                params[which] += (mu_max - 10E3)  *dnest4.randh()
                params[which] = dnest4.wrap(params[which], 10E3, mu_max)
            elif velo_which ==1: #ln(1/beta)
                space = np.log(1/.1)
                params[which] += space *dnest4.randh()
                params[which] = dnest4.wrap(params[which], 0, np.log(1/.1))
            elif velo_which == 2:
                minval, maxval = 5E6, 1E8
                params[which] += (maxval-minval) *dnest4.randh()
                params[which] = dnest4.wrap(params[which], minval, maxval)

        elif which == trap_idx:
          params[which] += prior_vars[trap_idx]*dnest4.randh()
          params[which] = dnest4.wrap(params[which], 50, 1000)

        else: #velocity or rc params: cant be below 0, can be arb. large
            print "which value %d not supported" % which
            exit(0)


        return logH

    def log_likelihood(self, params):
        return self.mpi_manager.calc_likelihood(params)

    def get_indices(self):
        return (tf_first_idx, velo_first_idx, trap_idx, grad_idx, len(priors))
