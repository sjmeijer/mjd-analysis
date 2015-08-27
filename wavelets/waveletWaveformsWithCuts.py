#!/usr/bin/python
from ROOT import *
TROOT.gApplication.ExecuteFile("$MGDODIR/Root/LoadMGDOClasses.C")
TROOT.gApplication.ExecuteFile("$MGDODIR/Majorana/LoadMGDOMJClasses.C")
import sys
import array
from ctypes import c_ulonglong
import os

import pylab
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy import ndimage, signal
from skimage.filters import threshold_otsu, rank
import mallat_wavelets
from skimage.morphology import reconstruction
from scipy.ndimage import gaussian_filter
from skimage.morphology import disk
from scipy.stats import scoreatpercentile



"""
    Makes cuts based on gatified data, plots the waveforms from the built data
    
    """



channelToPlot = 4
energyCut = "energyCal>%f && energyCal < %f" % (2610, 2620);
chanCut = "channel == 68";

energyName = "energyCal"
currentName = "rcDeriv50nsMax"
aeCut = 0.0485011
aeScale = -0.00000050242948

aePassStr = "abs(%s/%s) -(%.15f*(%s)) > %.7f" % (currentName, energyName, aeScale, energyName, aeCut)
aeCutStr = "abs(%s/%s) -(%.15f*(%s)) < %.7f" % (currentName, energyName, aeScale, energyName, aeCut)


flatTimeSamples = 300

#Instantiate and prepare the baseline remover transform:
baseline = MGWFBaselineRemover()
baseline.SetBaselineSamples(flatTimeSamples)

rcdiff = MGWFRCDifferentiation()
rcdiff.SetTimeConstant(50 * CLHEP.ns )


def main(argv):
    if len(argv) < 2:
        print 'Usage: waveletWaveformsWithCuts.py [Built File] [Gatified File]'
        sys.exit()

    chain = TChain("MGTree")
    chain.SetDirectory(0)
    chain.Add( sys.argv[1] )

    chain2 = TChain("mjdTree")
    chain2.SetDirectory(0)
    chain2.Add( sys.argv[2] )

#    #set up a canvas :
#    canvas = TCanvas("canvas")


    # Define your cuts
    #energyCut = "energy>%f && energy<%f" % (274*10**3,276*10**3);

    cut = energyCut + " && " + chanCut
    print "The cuts will be: " + cut

    chain.AddFriend(chain2,"mjdTree")
    chain2.SetEntryList(0)
    chain2.Draw(">>elist", cut, "entrylist")
    elist = gDirectory.Get("elist")
    print "Number of entries in the entryList is " + str(elist.GetN())
    chain2.SetEntryList(elist);

    energyCalVec = vector("double")()
#chanVec = vector("double")()
    aVec = vector("double")()

    plt.ion()
    fig = plt.figure(1)
    fig2 = plt.figure(2)
    fig3 = plt.figure(3)
    fig4 = plt.figure(4)

    for ientry in xrange( elist.GetN() ):
        entryNumber = chain2.GetEntryNumber(ientry);
        chain.LoadTree( entryNumber )
        chain.GetEntry( entryNumber )
        
        aVec = chain.rcDeriv50nsMax
        energyCalVec = chain.energyCal
        
        # tree contains MGTEvent objects
        event = chain.event
        #run   = chain.run
        # Two loops because with event building you can have 1+ WF per event, due to the event grouping
        for i_wfm in xrange( event.GetNWaveforms() ):
            
            sisData   = event.GetDigitizerData( i_wfm )
            channel   = sisData.GetChannel()
            if (channel != channelToPlot): #148
                continue
        
            #eventTime = sisData.GetTimeStamp()
            #runNumber = run.GetRunNumber()
            wf = event.GetWaveform(i_wfm)
            
            baseline.TransformInPlace(wf)
            
            ext = MGWFExtremumFinder()
            ext.SetFindMaximum(1)
            ext.TransformInPlace(wf)
            wfmax = ext.GetTheExtremumValue()
            

            print('>>>Event:%s Waveform: %s' % (ientry, i_wfm))
            
            a = aVec.at(i_wfm)
            e = energyCalVec.at(i_wfm)
            
            aePass = ((a /e ) + aeScale*e) > aeCut
            
            if aePass:
                print ">>>A/E says this is single site"
            else:
                print ">>>A/E says this is multi site"

            orig_length = wf.GetLength()
            wflength = 2**10
            wf.SetLength(wflength)

#            wf.SetLength(wflength)
            for i in xrange(wflength-orig_length):
                at_sizet = vector("size_t")(1)
                at_sizet[0] = int((orig_length-i)-1)
                #print orig_length-i
                wf.SetValue(i+orig_length, wf.At(at_sizet[0]))
            
            np_data = wf.GetVectorData()
            
            
            nvoice = 128.;
            oct = 5.
            scale = 64.
            gaussian_smoothing =2

            np_data_smoothed = np_data
            #np_data_smoothed=ndimage.filters.gaussian_filter1d(np_data, 2)

            
            plt.figure(fig.number)
            plt.clf()
#            fig.canvas.clear()



#           rwt = plotRWT(np_data_smoothed, nvoice, oct, scale)

#fig.canvas.draw()
            
            #plotSkel(rwt, nvoice, oct, scale, fig2, fig3)
            ridges = plotSciPy(np_data_smoothed, nvoice, oct, scale, fig, fig2)
            
            plt.figure(fig4.number)
            plt.clf()
            plt.axis([350, 550, -0.1, 1])
            
            for ridge in ridges:
                plt.plot([ridge[1][0], ridge[1][0]], [0, 1])
            
            #Normalize for wf plots
            #np_data_smoothed/=wfmax
            wf/=wfmax
            origWFNorm = wf.GetVectorData()
            #plt.plot(np_data_smoothed)
            plt.plot(origWFNorm)

            plt.gca().set_xlabel('Time stamp number')
            plt.gca().set_ylabel('Normalized Charge')
            if aePass:
                plt.gca().set_title('A/E Single-Site')
            else:
                plt.gca().set_title('A/E Multi-Site')

            plt.figure(fig3.number)
            plt.clf()
            
            rcdiff.TransformInPlace(wf)
            ext.TransformInPlace(wf)
            wfmax = ext.GetTheExtremumValue()
            
            wf /= wfmax
            
            #plt.axis([350, 550, -0.1, 1])
            
            for ridge in ridges:
                plt.plot([ridge[1][0], ridge[1][0]], [0, 1])
            wcdiffdata = wf.GetVectorData()
            
            plt.plot(wcdiffdata)
            plt.axis([350, 550, -0.1, 1])
            plt.gca().set_xlabel('Time stamp number')
            plt.gca().set_ylabel('Normalized Current')

            if aePass:
                plt.gca().set_title('A/E Single-Site')
            else:
                plt.gca().set_title('A/E Multi-Site')

            #plt.plot(np_data_smoothed-np_data)
            

#            if numWfs != 1:
            value = raw_input('  --> Press q to quit, any other key to continue\n')

            if value == 'q':
                exit(1)


    exit(1)

def plotSciPy(data, nvoice, oct, scale, fig0, fig1):
    #save for plotting later
    #rwt = np.copy(rwt)
    
    wavelet = 'DerGauss';
    rwt = mallat_wavelets.rwt(data,nvoice,wavelet,oct,scale)
    
    xlimits = [350,550]
    

    #calcs
    rwt = np.flipud( np.transpose(rwt) )
    im_rwt = np.copy(rwt)
    
    im_rwt[:,0:xlimits[0]] = 0
    im_rwt[:,xlimits[1]::] = 0
    
    (nscale,n) = rwt.shape;
    
    max_distances = np.empty(n)
    max_distances[:]=1
    
    gap_thresh = 10
    
    ridges = mallat_wavelets.scipy_ridges(rwt, max_distances, gap_thresh)
    filteredRidges = mallat_wavelets.filter_ridge_lines(rwt, ridges, min_length = 10)
    
    ridgesToPlot = ridges
    
    #plot the RWT
    ############################################################

    plt.figure(fig0.number)
    plt.clf()
    
    #
#    im_rwt /= np.max(im_rwt)

    #normalize scale by scale for the plot
    for k in xrange(nscale):
        #print "nscale " + str(nscale) + ", shape im_rwt " + str(im_rwt.shape) + ", k " + str(k)
        amax  = np.amax(im_rwt[k,xlimits[0]:xlimits[1]]);
        amin = np.amin(im_rwt[k,xlimits[0]:xlimits[1]]);
        im_rwt[k,:] = np.log10( ((im_rwt[k,:])-amin) / (amax-amin) *2048 +1)
    
    
    plt.imshow( im_rwt , cmap='Greys', aspect='auto', origin='lower',extent=[0,n,0,nscale])
    fig0.gca().set_title("Wavelet power, normalized at each scale to the scale max")
    
#    fig0.canvas.draw()

    minscale = 2
    
    #plot the ridges
    for ridge in ridgesToPlot:
        plt.plot(ridge[1],ridge[0]);
    
    plt.gca().invert_yaxis()
    plt.axis([350, 550,nscale,0])

    plt.gca().set_xlabel('Time stamp number')
    plt.gca().set_ylabel('log2(Wavelet scale) idx')

    cb = plt.colorbar()

#plt.axis([350, 550, 0, nscale])

    fig0.canvas.draw()

############################################################
    
    plt.figure(fig1.number)
    plt.clf()
    
    for ridge in ridgesToPlot:
        plt.plot(ridge[0],np.log2(rwt[ ridge[0],  ridge[1]]) );
    #plt.xticks(ytix)

    plt.gca().set_ylabel('log2(Wavelet power)')
    plt.gca().set_xlabel('log2(Wavelet scale) idx')


#plt.gca().set_xrange(350,550)

#plt.xlim(minscale, minscale+oct);


    fig1.canvas.draw()

    
    return ridgesToPlot


def plotSkel(rwt, nvoice, oct, scale, fig1, fig2):
    maxmap = mallat_wavelets.MM_RWT(rwt, 10);
    
#    print "size of maxmap is " + str(maxmap.shape)
#    maxmap[0:200,:] = 0
#    maxmap[800::,:] = 0

    (skellist,skelptr,skellen) = mallat_wavelets.SkelMap(maxmap);

    #set(gcf, 'NumberTitle','off', 'Name','Window 2')
    l = len(skelptr)
    
    (n,nscale) = rwt.shape;
    minscale = 2
    
    plt.figure(fig1.number)
    plt.clf()
    
    

    ridges = mallat_wavelets.PlotSkelMap(n,scale,skellist,skelptr[0:l],skellen[0:l],'','b',[],nvoice,minscale,oct, rwt);
    
    plt.gca().set_xlabel('Time stamp number')
    plt.gca().set_ylabel('log2(Wavelet scale)')
    
    fig1.canvas.draw()
    
    plt.figure(fig2.number)
    plt.clf()
    
    
    for i in range(len(ridges)):
        ridge = mallat_wavelets.ExtractRidge(ridges[i],rwt,skellist,skelptr,skellen,oct,scale)
        
        plt.plot(ridge[:,0],ridge[:,1]);
    #plt.xticks(ytix)
    #plt.gca().invert_xaxis()

    plt.gca().set_ylabel('Wavelet power')
    plt.gca().set_xlabel('log2(Wavelet scale)')

    #plt.xlim(minscale, minscale+oct);


    fig2.canvas.draw()


    return


def plotRWT(x, nvoice, oct, scale):

    
    wavelet = 'DerGauss';

    rwt = mallat_wavelets.rwt(x,nvoice,wavelet,oct,scale)
    
    
    #kill the edges
    
    #rwt = rwt[200:800, :]


    
    #rwt = sp.ndimage.filters.gaussian_filter(rwt, 5)

#print "size of rwt is " + str(rwt.shape)

    logrwt = np.transpose(rwt);
    logim = np.flipud(logrwt);
    #print "size of logim is " + str(logim.shape)

    linrwt = mallat_wavelets.log2lin(logim,nvoice);
    #print "size of linrwt is " + str(linrwt.shape)

#    delta = 1./15;
#    unit = (1-3.*delta)/3;
#    h2 = [delta, delta, 1-2*delta, 2*unit];
#    h1 = [delta, 2*(unit+delta), 1-2*delta, unit];

    (n, nscale) = linrwt.shape



#    plt.imshow( linrwt , cmap='gray_r')
#    
#
#    #plt.axes('position',h1);
#    plt.axis([1, len(x), np.floor(min(x)), np.floor(max(x))])
#    ##plt.axes('position',h2)


    
    im_rwt = np.copy(rwt)
    (n, nscale) = im_rwt.shape
#    
#    im_rwt /= np.max(im_rwt)

    for k in xrange(nscale):
        #print "nscale " + str(nscale) + ", shape im_rwt " + str(im_rwt.shape) + ", k " + str(k)
        amax  = np.amax(im_rwt[:,k]);
        amin = np.amin(im_rwt[:,k]);
        im_rwt[:,k] = ((im_rwt[:,k])-amin) / (amax-amin) *1;
    
    

#    radius = 5
#    selem = disk(radius)
#    local_otsu = rank.otsu(im_rwt, selem)
#
#    skel = skeletonize(im_rwt)

#    im_rwt = gaussian_filter(im_rwt, 1)
#    seed = np.copy(im_rwt)
#    seed[1:-1, 1:-1] = im_rwt.min()
#    mask = im_rwt
#    im_rwt = reconstruction(seed, mask, method='dilation')

#    sx = ndimage.sobel(im_rwt, axis=0, mode='constant')
#    sy = ndimage.sobel(im_rwt, axis=1, mode='reflect')
#    im_rwt = np.hypot(sx, sy)

#    mask = np.zeros((n,nscale))
#    
#    for k in range(nscale):
#        mask[:,k] = im_rwt[:,k] > .05*np.amax(im_rwt[:,k])

#    val = filter.threshold_otsu(im_rwt)
#    mask = im_rwt < val
#    im_rwt, nb_labels = ndimage.label(mask)
#    print "There are %d regions" % nb_labels

#ybounds   = [(2+(oct-np.floor(np.log2(scale)))), np.log2(n*2/scale)];

    minOct = oct
    maxOct = np.floor(np.log2(n))
    
    minScale = pow(2,minOct) * scale
    maxScale = pow(2,maxOct) * scale

#print "minscale %f, maxscale %f: " % (np.floor(np.log2(minScale)),  np.floor(np.log2(maxScale) ) )
    
    ybounds   = [ 2+(oct-np.floor(np.log2(scale))),np.log2(n)+2-np.floor(np.log2(scale)) ];
    
    #plt.imshow( np.flipud( np.transpose(im_rwt) ) , cmap='gray_r', origin='lower',extent=[0,n,ytix[0],ytix[1]], aspect='auto')
    #plt.imshow( np.flipud( np.transpose(im_rwt) ) , cmap='gray_r', origin='lower',extent=[0,n,ytix[0],ytix[1]], aspect='auto')
    plt.imshow( np.flipud( np.transpose(im_rwt ) ) , cmap='gist_heat', origin='lower', extent=[0,n,ybounds[0],ybounds[1]],aspect='auto')
    #plt.imshow(np.flipud( np.transpose(dist_on_skel) ), cmap=plt.cm.spectral, interpolation='nearest',aspect='auto')
    #plt.yticks(ytix)
    plt.gca().invert_yaxis()

    #mallat_wavelets.ImageRWT( np.fliplr( np.transpose(lin(rwt) ),'Individual','gray','lin',3,16)
    cb = plt.colorbar()
    
    plt.gca().set_xlabel('Time stamp number')
    plt.gca().set_ylabel('log2(Scale)')
    
    
    return rwt

    #plt.axis('ij')

#plt.show()




def plotOld(data, w, title):
    w = pywt.Wavelet(w)
    a = data
    ca = []
    cd = []
    numLevels = 10
    mode = pywt.MODES.sym
    DWT = 1
    
    if DWT:
        for i in xrange(numLevels):
            (a, d) = pywt.dwt(a, w, mode)
            ca.append(a)
            cd.append(d)
    else:
        coeffs = pywt.swt(data, w, numLevels)  # [(cA5, cD5), ..., (cA1, cD1)]
        for a, d in reversed(coeffs):
            ca.append(a)
            cd.append(d)

    pylab.figure()
    ax_main = pylab.subplot(len(ca) + 1, 1, 1)
    pylab.title(title)
    ax_main.plot(data)
    pylab.xlim(0, len(data) - 1)
    
    for i, x in enumerate(ca):
        ax = pylab.subplot(len(ca) + 1, 2, 3 + i * 2)
        ax.plot(x, 'r')
        if DWT:
            pylab.xlim(0, len(x) - 1)
        else:
            pylab.xlim(w.dec_len * i, len(x) - 1 - w.dec_len * i)
        pylab.ylabel("A%d" % (i + 1))

    for i, x in enumerate(cd):
        ax = pylab.subplot(len(cd) + 1, 2, 4 + i * 2)
        ax.plot(x, 'g')
        pylab.xlim(0, len(x) - 1)
        if DWT:
            pylab.ylim(min(0, 1.4 * min(x)), max(0, 1.4 * max(x)))
        else:  # SWT
            pylab.ylim(
                       min(0, 2 * min(
                                      x[w.dec_len * (1 + i):len(x) - w.dec_len * (1 + i)])),
                       max(0, 2 * max(
                                      x[w.dec_len * (1 + i):len(x) - w.dec_len * (1 + i)]))
                       )
        pylab.ylabel("D%d" % (i + 1))



if __name__=="__main__":
    main(sys.argv[1:])

