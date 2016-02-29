import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

import toy_model as tm
import pymc
#plt.style.use('presentation')

def main():

    #Randomly generate a signal to fit to
  
    true_early_mean = 0
    true_late_mean = 2
    true_early_sigma = 1
    true_late_sigma = 0.75
    switchpoint = 133

    early_signal = np.random.normal(true_early_mean, true_early_sigma, switchpoint)
    late_signal = np.random.normal(true_late_mean, true_late_sigma, 200-switchpoint)

    signal = np.append(early_signal, late_signal)
    
    #plot that shit
    fig10 = plt.figure(10)
    plt.plot( np.arange(0,100),0.5*np.ones(100)  ,color="blue" )
    plt.plot( np.arange(100,200),0.5*np.ones(100)  ,color="red" )
    plt.axvline(100  ,color="black", linestyle=":" )
    plt.savefig("prior.pdf")
    
    
    #set up a pymc model
    siggen_model = pymc.Model( tm.createSignalModel(signal) )
    M = pymc.MCMC(siggen_model)
    
    #Change the proposal distribution from default (prior) to a normal distribution for t0
    #And make t_0 discrete
    M.use_step_method(pymc.DiscreteMetropolis, M.switchpoint, proposal_sd=4., proposal_distribution='Normal')
    M.use_step_method(pymc.Metropolis, M.early_mu, proposal_sd=0.1, proposal_distribution='Normal')
    M.use_step_method(pymc.Metropolis, M.late_mu, proposal_sd=0.1, proposal_distribution='Normal')
    M.use_step_method(pymc.Metropolis, M.early_sigma, proposal_sd=0.1, proposal_distribution='Normal')
    M.use_step_method(pymc.Metropolis, M.late_sigma, proposal_sd=0.1, proposal_distribution='Normal')

    #Run the MCMC
    M.sample(iter=10000)

    #how long do you wait to converge before you decide you're really sampling your posterior?
    burnin = 4000

    #pull out the fit parameters after burnin
    t0 = np.median(M.trace('switchpoint')[burnin:])
    early_mu = np.median(M.trace('early_mu')[burnin:])
    late_mu = np.median(M.trace('late_mu')[burnin:])
    print "t0 = %f pm %f" % (t0, np.std(M.trace('switchpoint')[burnin:]))


    #Make a bunch of plots

    early_mu_arr = M.trace('early_mu')[burnin:]
    early_sigma_arr = M.trace('early_sigma')[burnin:]
    
    print "early_mu = %f pm %f" % (early_mu, np.std(M.trace('early_mu')[burnin:]))
    print "early_sig = %f pm %f" % (np.mean(M.trace('early_sigma')[burnin:]), np.std(M.trace('early_sigma')[burnin:]))
    
    n_bins = 50
    
    #Histograms of posteriors
    
    fig0 = plt.figure(0)
    weights = np.ones_like(early_mu_arr)/float(len(early_mu_arr))
    n, bins, patches = plt.hist(early_mu_arr, n_bins, histtype='step', linewidth=5, weights=weights)
    plt.xlabel("mu_0 value")
    plt.ylabel("probability")
    plt.savefig("early_mu_pdf.pdf")
    
    fig0 = plt.figure(6)
    weights = np.ones_like(early_sigma_arr)/float(len(early_sigma_arr))
    n, bins, patches = plt.hist(early_sigma_arr, n_bins, histtype='step', linewidth=5, weights=weights)
    plt.xlabel("sigma_0 value")
    plt.ylabel("probability")
    plt.savefig("early_sigma_pdf.pdf")

    #recreate the best-fit sample to plot it
    e = np.ones(t0)*early_mu
    l = np.ones(200-t0)*late_mu
    s = np.append(e,l)

    #What's your (randomly generated) data look like?
    fig = plt.figure(1)
    plt.clf()
    plt.plot( signal  ,color="red" )
    plt.savefig("toy_data.pdf")
    
    #What's your fit look like?
    fig2 = plt.figure(2)
    plt.plot( s  ,color="blue" )
    plt.plot( signal  ,color="red" )
    plt.savefig("toy_fit.pdf")

    #MCMC full trace
    f, axarr = plt.subplots(5, sharex=True)
    axarr[0].plot(M.trace('switchpoint')[:])
    axarr[4].set_xlabel('MCMC Step Number')
    axarr[0].set_ylabel('t_0')
    axarr[1].plot(M.trace('early_mu')[:])
    axarr[1].set_ylabel('mu_0')
    axarr[2].plot(M.trace('late_mu')[:])
    axarr[2].set_ylabel('mu_1')
    axarr[3].plot(M.trace('early_sigma')[:])
    axarr[3].set_ylabel('sigma_0')
    axarr[4].plot(M.trace('late_sigma')[:])
    axarr[4].set_ylabel('sigma_1')
    plt.savefig("MCMC_steps.pdf")
    
    #Zoom in to see the convergence
    f, axarr = plt.subplots(5, sharex=True)
    axarr[0].plot(M.trace('switchpoint')[:])
    axarr[4].set_xlabel('MCMC Step Number')
    axarr[0].set_ylabel('t_0')
    axarr[1].plot(M.trace('early_mu')[:])
    axarr[1].set_ylabel('mu_0')
    axarr[2].plot(M.trace('late_mu')[:])
    axarr[2].set_ylabel('mu_1')
    axarr[3].plot(M.trace('early_sigma')[:])
    axarr[3].set_ylabel('sigma_0')
    axarr[4].plot(M.trace('late_sigma')[:])
    axarr[4].set_ylabel('sigma_1')
    plt.xlim(0,1000)
    plt.savefig("MCMC_steps_zoom.pdf")

    plt.show()




if __name__=="__main__":
    main()