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
import pywt
import mallat_wavelets


"""
    Makes cuts based on gatified data, plots the waveforms from the built data
    
    """



channelToPlot = 4
energyCut = "energy>%f" % (100*10**3);
chanCut = "channel == 68";

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

    #set up a canvas :
    canvas = TCanvas("canvas")


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

    plt.ion()
    fig = plt.figure(1)
    fig2 = plt.figure(2)


    for ientry in xrange( elist.GetN() ):
        entryNumber = chain2.GetEntryNumber(ientry);
        chain.LoadTree( entryNumber )
        chain.GetEntry( entryNumber )
        
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
            
            ext = MGWFExtremumFinder()
            ext.SetFindMaximum(1)
            ext.TransformInPlace(wf)
            max = ext.GetTheExtremumValue()
            wf/=max

            print('>>>Event:%s Waveform: %s' % (ientry, i_wfm))
            
            wfHist = wf.GimmeHist().Clone()
            textsize = 0.06;
            wfHist.SetTitle("");
            wfHist.GetXaxis().SetTitle('Time [ns]')
            wfHist.GetYaxis().SetTitle('Voltage [a.u.]')
            #wfHist.GetYaxis().SetTitleOffset(1.4);
            wfHist.GetYaxis().CenterTitle()
            wfHist.GetXaxis().CenterTitle()
            wfHist.GetXaxis().SetTitleSize(textsize)
            wfHist.GetYaxis().SetTitleSize(textsize)
            wfHist.SetStats(false);
            wfHist.SetLineColor(2)
            gPad.SetMargin(0.15, 0.1, 0.15, 0.1)
            wfHist.Draw()
            canvas.Update()
            

            

            
            wflength = 2**10
            wf.SetLength(wflength)
            np_data = wf.GetVectorData()
            
#            np_data = np.empty(wflength)
#            
#            start_offset = np.floor((wf.GetLength() - wflength) * .5)
#            waveform_vector = wf.GetVectorData()
#            
#            for i in xrange(wflength):
#                np_data[i] = waveform_vector.at(int(i+start_offset))

            #print np_data[0:10]
            
            plt.figure(fig.number)
            plt.clf()
#            fig.canvas.clear()

            rwt = plotRWT(np_data)
            fig.canvas.draw()
            
            plt.figure(fig2.number)
            plt.clf()
            numWfs = plotSkel(rwt)
            fig2.canvas.draw()
#

#            if numWfs != 1:
            value = raw_input('  --> Press q to quit, any other key to continue')

            if value == 'q':
                exit(1)
#if value == 'p':
                

                #plot(waveform_vector, 'haar', "Haar: Signal irregularity shown in Haar wavelet")
                #pylab.show()

    exit(1)

def plotSkel(rwt):
    maxmap = mallat_wavelets.MM_RWT(rwt, 1000);
    (skellist,skelptr,skellen) = mallat_wavelets.SkelMap(maxmap);

    #set(gcf, 'NumberTitle','off', 'Name','Window 2')
    l = len(skelptr)
    
    n = 2**10
    nvoice = 12.
    oct = 3.
    scale = 128.
    minscale = 2

    numWfs = mallat_wavelets.PlotSkelMap(n,scale,skellist,skelptr[0:l],skellen[0:l],'','b',[],nvoice,minscale,oct, rwt);

    return numWfs


def plotRWT(x):
    nvoice = 12.;
    oct = 3
    scale = 128
    
    wavelet = 'DerGauss';

    rwt = mallat_wavelets.rwt(x,nvoice,wavelet,oct,scale)

    print "size of rwt is " + str(rwt.shape)

    logrwt = np.transpose(rwt);
    logim = np.flipud(logrwt);
    print "size of logim is " + str(logim.shape)

    linrwt = mallat_wavelets.log2lin(logim,nvoice);
    print "size of linrwt is " + str(linrwt.shape)

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


    
    im_rwt = np.fliplr( np.transpose(linrwt) )
    
    (n, nscale) = im_rwt.shape
    
#    for k in xrange(nscale):
#        #print "nscale " + str(nscale) + ", shape im_rwt " + str(im_rwt.shape) + ", k " + str(k)
#        amax  = np.amax(im_rwt[:,k]);
#        amin = np.amin(im_rwt[:,k]);
#        im_rwt[:,k] = ((im_rwt[:,k])-amin) / (amax-amin) *256;

    ytix   = [pow(2, (2+(oct-np.floor(np.log2(scale))))),n*2/scale];
    plt.imshow( np.flipud( np.transpose(im_rwt) ) , cmap='gray_r', origin='lower',extent=[0,n,ytix[0],ytix[1]], aspect='auto')
    #plt.yticks(ytix)
    plt.gca().invert_yaxis()

    #mallat_wavelets.ImageRWT( np.fliplr( np.transpose(linrwt) ),'Individual','gray','lin',3,16)
    cb = plt.colorbar()

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

