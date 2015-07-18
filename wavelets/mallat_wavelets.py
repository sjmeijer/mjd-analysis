import numpy as np
import sys
import matplotlib.pyplot as plt
import pylab

def rwt(x,nvoice,wavelet,oct=2,scale=4):
#%  Usage
#%    rwt(x,nvoice,wavelet)
#%  Inputs
#%    x        signal, dyadic length n=2^J, real-valued
#%    nvoice   number of voices/octave (ie, your resolution)
#%    wavelet  string 'Gauss', 'DerGauss','Sombrero', 'Morlet'
#     oct      lowest octave which will be addressed
#     scale    the starting scale factor, which will be modified by the octave
#              should be a 2^n int
#%  Outputs
#%    rwtMat    matrix n by nscale where
#%             nscale = nvoice .* noctave
#%
#%  Description
#%     see sections 4.3.1 and 4.3.3 of Mallat's book

    #preparation
    #x = ShapeAsRow(x);
    n = len(x);
    
    if not is_power2(n):
        print "Data length is %d, must be 2^n" % n
        exit(1)
    
    xhat = np.fft.fft(x);

    xi   = np.linspace(0, n/2., n/2.+1)
    xi2 = np.linspace(-n/2.+1, -1, n/2.-1)
    xi = np.concatenate((xi,xi2)) * (2.*np.pi/n)

#    print "size of xi: " + str (xi.shape)
    #root
    #omega0 = 5.;

    #largest octave available is set by the length of your dataset...
    noctave = np.floor(np.log2(n))-oct;
    #so I can loop from the highest scale [2^log2(n)] to the lowest scale [2^log2(oct)]

    nscale  = nvoice * noctave;

    print "nscale will be %d, noctave %d" % (nscale, noctave)
    rwtMat = np.zeros((n,nscale));

    kscale  = 0;

    for jo in  range(int(noctave)):
        #scale here will be the initial scale * 2^jo
        oct_scale = scale * pow(2, jo)
        for jv in range(1, int(nvoice)+1 ):
            qscale = oct_scale * pow(2, float(jv)/nvoice);
            
            #print "scale %f, nvoice %f, jo %d, jv %f, (qscale) %f" % (oct_scale, nvoice, jo, jv, (qscale))
            
            omega =  n * xi / qscale ;
            #print "omega: " + str(omega[0:5])
            if (wavelet == 'Gauss'):
                window = np.exp( -np.square(omega) / 2. );
            elif (wavelet == 'DerGauss'):
                window = omega *np.exp(-np.square(omega) /2.) * 1j;
            elif (wavelet =='Sombrero'):
                window = ( np.square(omega) ) * np.exp(-np.square(omega) /2.);
#            elif strcmp(wavelet,'Morlet'):
#                window = exp(-(omega - omega0).^2 ./2) - exp(-(omega.^2 + omega0.^2)/2);

            #Renormalization by scale
            window = window / np.sqrt(qscale);
            what = window * xhat;
            w    = np.fft.ifft(what);
                
#            print "size of w: " + str (w.shape)
#            print "size of rwtMat: " + str (rwtMat.shape)

            rwtMat[ :,kscale] = np.transpose( np.real(w) );

            kscale = kscale+1;

    return rwtMat

def log2lin(logim,nvoice=1.):
#%
#% function linim = log2lin(logim,nvoice);
#%
#% Description : takes a log-scale image l , and transforms it into
#%               a linear scale image n
#%
#% Inputs :	logim : logscale image
#%              	nvoice : number of voices (default = 1)
#%
#% Outputs :	linim : linear scale image
#%
#% See also : CWT, ImageCWT
#%
    
    (mn,n) = logim.shape
    
    #print "mn %d, n %d" % (mn, n)
    
    m= mn/nvoice; #this is the number of octaves
    s0 = n/(pow(2,m));
    j0 = np.log2(s0);
    s1 = n;
    max = np.floor(n/ 2**(1./nvoice) );
    
    print "s0 %f, j0 %f, max %f" % (s0, j0, max)
    
#% normally, linim has size (max-s0+1,n);
#% but this is too big (out of memory for n = 1024)
#% so we replace it by (floor((max-s0+1)/10),n);
#
#%   sz = floor((max-s0)/10)+1;

    sz = np.floor((max-s0)/1.)+1;
    linim = np.zeros((sz,n));
    
    #print "n %d, nvoice %d, max %d, sz %d, s0 %d, j0 %d" % (n, nvoice, max, sz, s0, j0)
    
    for j in xrange(int(sz)):
        i = (j) + s0;
        li = (np.log2(i)-j0) * float(nvoice);
        fli = np.floor(li);
        rli = li - fli;
        #print "shape of logim is " + str(logim.shape)
        #print "j=%d, fli %d, rli %f" % (j, fli, rli)

        if not (rli == 0):
            linim[j,:] = rli * logim[fli+1,:];
        if not (rli == 1):
            linim[j,:] =linim[j,:]+(1.-rli) * logim[fli,:];
    return linim

def ImageRWT(rwt,scaling,colors,option='log',oct=2,scale=4):
#% ImageRWT -- Image of Continuous Wavelet Transform
#%  Usage
#%    ImageRWT(rwt,scaling,colors)
#%  Inputs
#%    rwt      matrix produced by RWT
#%    scaling  string, 'Overall', 'Individual'
#%    colors   string  argument for colormap
#%    option   'lin' or 'log' for the type of display
#%
#%  Side Effects
#%    Image Display of Continuous Wavelet Transform
#%

    (n, nscale) = (rwt.shape);
    noctave= np.floor(np.log2(n)) -2;
    nvoice = nscale / noctave;

    if  (option == 'log'):
        ytix   = np.linspace(2+(oct-np.floor(np.log2(scale))),np.log2(n)+2-np.floor(np.log2(scale)),nscale);
        xtix   = np.linspace(0,n,n);
    else:
        xtix   = np.linspace(0,n,n);
        ytix   = np.linspace( pow(2, (2+(oct-np.floor(np.log(scale)))))  ,n*2./scale,nscale);

    if (scaling == 'Individual'):
        for k in xrange( int(nscale)):
            amax  = max(rwt[:,k]);
            amin = min(rwt[:,k]);
            rwt[:,k] = ((rwt[:,k])-amin) / (amax-amin) * 256.;
    else:
        amin = min(min(rwt));
        amax = max(max((rwt)));
        rwt = (rwt+amax) / (2.*amax) *256.;

    if (option=='lin'):
        plt.imshow(xtix,ytix, np.flipud(np.transpose(rwt)));
        plt.axis('xy');
    else:
        plt.imshow(xtix,ytix, np.transpose(rwt));
        axis('ij');
        xlabel('')
        ylabel('log2(s)')
    str = sprintf('colormap(1-%s(256))',colors);
    eval(str)


def MM_RWT(rwt,par=1000):
#% MM_RWT -- Modulus Maxima of a Real Wavelet Transform
#%  Usage
#%    maxmap = MM_RWT(rwt,par)
#%  Inputs
#%    rwt    Output of RWT
#%    par    optional. If present, keep thresholds only
#%           above a certain value. default = 1000
#%  Outputs
#%    maxmap binary array indicating presence of max or not
#%
#%  Description
#%    Used to calculate fractal exponents etc.
#%

    (n, nscale) = rwt.shape;

    
    maxmap = np.zeros((n, nscale));
    
    t      = range(n);
    tplus  = np.roll(t, 1)
    tminus = np.roll(t, -1)
    
    #only look for maxes, don't look for mins
    #rwt    = np.abs(rwt);
    
    for k in xrange( int(nscale) ):
        
        #are you higher than t-1 and t+1?
        localmax =  np.logical_and( np.greater(rwt[t,k], rwt[tplus,k]), np.greater(rwt[t,k], rwt[tminus,k]) )
        
        #find the rwt value for the maxes (by multiplying by the boolean localmax matrix)
        y =  np.multiply(localmax[:], rwt[:,k]);

        #print "localmax size " + str(localmax.shape) + " y size + " + str(y.shape)
        maxy = np.amax(y);
            #print "maxy " + str(maxy)
        maxmap[t,k] = (y >= maxy/par);

    return maxmap

def SkelMap(maxmap):
#% SkelMap -- Chain together Ridges of Wavelet Transform
#%  Usage
#%    [skellist,skelptr,skellen] = SkelMap(maxmap)
#%  Inputs
#%    maxmap    matrix MM_RWT
#%  Outputs
#%    skellist  storage for list of chains
#%    skelptr   vector of length nchain --pointers
#%              to head of chain
#%    skellen   vector of length nchain -- length of skellists
#%
#%  Description
#%    A chain is a list of maxima at essentially the same position
#%    across a range of scales.
#%    It is identified from the maxmap data structure output by WTMM
#%    by finding a root at coarse scales and identifying the closest
#%    maxima at the next finest scale.
#%    NO PROVISION IS MADE FOR 'terminating' A CHAIN before the
#%    finest scale is reached.
#%
#%    nchain = len(skellen) chains are found.
#%    A chain data structure is a list of scale-location pairs
#%    All chains are stored together in skellist.
#%    The k-th list begins in skellist at skelptr(k)
#%    The k-th list has length skellen(k)
#%
#%  See Also
#%    RWT, MM_RWT, PlotSkelMap, ExtractRidge
#%

    (n,nscale) = (maxmap.shape);
    noctave = np.floor(np.log2(n))-5;
    nvoice  = nscale/noctave;
    
    nchain = 0;
    chains = np.zeros(maxmap.shape);
    count  = 0;
    
    (a,b) = np.nonzero(maxmap)
    #print "maxmap nonzero size " + str(len(a))
    
    while np.any(np.any(maxmap)): #start new chain
        (iNonzero, jNonzero) = np.nonzero(maxmap)

        ind = np.lexsort((iNonzero, jNonzero))

#        print "ind " + str(ind[0:15])

        iNonzero = iNonzero[ind]
        jNonzero = jNonzero[ind]

#        print "i " + str(iNonzero[0:15])
#        print "j" + str(jNonzero[0:15])

        iscale = jNonzero[0];
        ipos   = iNonzero[0];
        #print "starting new chain #%d at %d,%d" % (nchain, iscale,ipos)
        
        chains[nchain,iscale] = ipos+1;
        maxmap[ipos,iscale] = 0;
        count = count+1;
                    
        while(iscale < nscale-1): # pursue rest of chain
            iscale = iscale+1;
#            print "iscale %d, nscale %d" % (iscale, nscale)
#            print np.nonzero(maxmap[:,iscale])
            j = np.nonzero(maxmap[:,iscale])[0];
            circdist   = np.amin( np.vstack( (np.abs(j-ipos),  np.abs(j-ipos+n), np.abs(j-ipos-n)) ), axis=0)
            #print circdist
            
            circdistdim = len(circdist.shape);
            
            pos = None
            
            if circdistdim == 1:
                
                if len(circdist) == 0:
                    continue
                
                #print "circdist is " + str(circdist.shape)
                (pos) = np.argmin(circdist);
                
                dist = circdist[pos]
                    #print "dist %d, pos is none" % (dist)
            else:
                #print "circdistdim is %d, I don't understand!  Crash!" % circdistdim
                sys.exit(0)
        
            if (pos is not None):
                ipos = j[pos];
                #print "nchain %d, iscale %d, ipos %d" % (nchain, iscale, ipos)
                chains[nchain,iscale] = ipos+1;
                #print('%i,%i\n',iscale,ipos,
                maxmap[ipos,iscale]   = 0;
                count = count+1;
            else:
                iscale = nscale;
        nchain +=1 ;
        # packed lists of chain structures
        
    rptr = np.zeros(n);
    rlen = np.zeros(n);
    pchain = 0;
    qchain = 0;
    store = np.zeros(2*count);
    
#print chains[0:10,0:10]
    
    for ch in xrange(nchain):
        rptr[ch] = pchain;
        j = np.nonzero(chains[ch,:])[0];
        
        iscale = j[0];
        rlen[ch] = len(j);
#        print j
#        print "chain %d at %d length %d" % (ch,pchain,rlen[ch])
#        print "iscale %d, rlen[ch] %d" % (iscale, rlen[ch])
        ix  = np.arange(int(iscale), int(iscale+rlen[ch]) );
        vec = np.vstack( (ix , chains[ch,ix] - 1) );
        
        #I think I want to get rid of the -1 here
        qchain = int(pchain + (2*rlen[ch]) -1);
        #print qchain
#        print "len of ix: " + str(len(ix))
#        print "shape of vec: " + str(vec.shape)
#        if ch == 50:
#            print "chain num is %d, looks to start at %d" % (ch, iscale)
#            print "this is vec row 2"
#            print ix[0:4]
#            print chains[ch,ix] - 1
#            print "later bro"


        vecFlat = vec.flatten(1)

#        print "len of vec flattened: " + str(vecFlat.shape)
#        print "pchain: %d, qchain %d" % (pchain, qchain)
#        
#        print "vecflat length: %d" % len(vecFlat[:])
#        print "store length: %d" % len(store)
        store[pchain:qchain+1] = vecFlat;
        pchain = qchain+1;
        
    skelptr  = rptr[0:nchain-1];
    skellen  = rlen[0:nchain-1];
    skellist = store[0:qchain-1];
    
    return (skellist, skelptr, skellen)

def PlotSkelMap(n,nscale,skellist,skelptr,skellen,
                titlestr='Skeleton of Wavelet Transform',color='y',
                chain=[],nvoice=12,minscale=2,noctave=-1, rwt=0):
#% PlotSkelMap -- Display Skeleton of Continuous Wavelet Transform
#%  Usage
#%    PlotSkelMap(n,nscale,skellist,skelptr,skellen [,titlestr,color,chain])
#%  Inputs
#%    n         signal length
#%    nscale    number of scales in cwt
#%    skellist  storage for list of chains
#%    skelptr   vector of length nchain -- pointers to heads of chains
#%    skellen   vector of length nchain -- length of skellists
#%    titlestr  optional, if number suppresses title string, if string
#%	       replaces default title string
#%    color    optional, if present specifies color of skeleton curves
#%              default is yellow.
#%    chain     optional, if present suppresses display of chain
#%              indicators
#%    nvoice    default=12
#%    minscale  default=2
#%    noctave   default=log2(n)-2
#%  Description
#%    A Time-Scale Diagram is drawn with the skeleton of the
#%    wavelet transform displayed
#%
#%  See Also
#%    BuildSkelMap, ExtractRidge
#%
    if noctave == -1:
        noctave = np.log2(n) - 2;

    nchain  = len(skelptr);

#    print skelptr[0:10]
#    print skellen[0:10]
#    print skellist[0:10]

    plt.axis((0, n, minscale, minscale+noctave));
    plt.gca().invert_yaxis()
#    axis('ij');
#    ylabel('log2(scale)');
#
#    if isstr(titlestr):
#        title(titlestr)
#    plotsymb = [color '-'];

    rwtMax = np.amax(np.amax( rwt ))
    maxLength = np.amax(skellen)

    passingRidges = []


    for k in range(int(nchain)):
        vec = np.zeros((2,skellen[k]));
        #print "skelptr %d, skellen %d" % (skelptr[k], skellen[k])
        ix  = np.arange( int(skelptr[k]), (int(skelptr[k] + 2*skellen[k])))
        #print "ix is " + str(ix)
        vecFlat = vec.flatten(1)
        vecFlat[:] = skellist[ix];
        
        #print vecFlat[0:20]
        vec = vecFlat.reshape((2,skellen[k]), order='F')
        
        pvec1 = vec[1,:]
        pvec2   = minscale+noctave - (vec[0,:]+1)/nvoice;
        
        
        #cut on length
        if skellen[k] < 0.1 * maxLength:
            continue
        
        #find the max value
        chainMax = 0
        
        for j in range(len(pvec1)):
            chainVal = rwt[pvec1[j], pvec2[j]]
            if chainVal > chainMax:
                chainMax = chainVal

#print "chainMax %f, rwtMax %f, percent %f" % (chainMax, rwtMax, (chainMax/rwtMax*100))
        if (chainMax/rwtMax < 0.1):
            #print "skipping chain %d of %d" % (k+1, nchain)
            continue
#        print skellist[ix]
#        print vec[1,:5]
#        print vec[0,:5]
#exit(1)
        #print vec.shape
                           
        plt.plot(pvec1,pvec2);
        passingRidges.append(k)

        
    plt.show()

    return passingRidges

def ExtractRidge(ridgenum,wt,skellist,skelptr,skellen,oct=2,sc=4):
#% ExtractRidge -- Pull One Ridge Continuous Wavelet Transform
#%  Usage
#%    ridge = ExtractRidge(ridgenum,wt,skellist,skelptr,skellen)
#%  Inputs
#%    ridgenum  index of ridge to extract, 0 <= ridgenum <= nchains -1
#%    wt        continuous wavelet transform output by CWT
#%    skellist  storage for list of chains
#%    skelptr   vector of length nchain -- pointers to heads of chains
#%    skellen   vector of length nchain -- length of skellists
#%  Outputs
#%	 ridge	   len by 2 array of numbers,
#%              each row is a scale, amplitude pair
#%
#%  Description
#%    The amplitude of the wavelet transform is followed along the
#%    ridge chain.
#%
#%  See Also
#%    CWT, WTMM, BuildSkelMap, PlotSkelMap
#%

    nchain  = len(skelptr);
    (n,nscale) = wt.shape;
    noctave = np.log2(n)-oct;
    nvoice  = nscale/noctave;
    
    if ridgenum < 0 or ridgenum > nchain:
        print 'ridge #%d not in range (0,%d)\n' %(ridgenum,nchain),
    
    head = skelptr[ridgenum];
    length  = skellen[ridgenum];
    
    ridge = np.zeros((length,2));


    vec = np.zeros((2,length));
    ix  = np.arange( int(head), (int(head + 2*length)))

    vecFlat = vec.flatten(1)
    vecFlat[:] = skellist[ix];
    vec = vecFlat.reshape((2,length), order='F')

#pvec2   = minscale+noctave - (vec[0,:]+1)/nvoice;
#print length
    for i in range(int(length)):
        iscale = vec[0,i];
        ipos   = vec[1,i];
        #scale  = (2 + oct - np.log2(sc) + iscale/nvoice);
        scale =  2 + noctave - (iscale+1)/nvoice;
        #amp    =  np.log2(np.abs((wt[ipos,iscale])));
        amp    =  (np.abs((wt[ipos,iscale])));
        ridge[i,0] = scale;
        ridge[i,1] = amp;

    return ridge
#def iconv(f,x):
#    #% iconv -- Convolution Tool for Two-Scale Transform
#    #%  Usage
#    #%    y = iconv(f,x)
#    #%  Inputs
#    #%    f   filter
#    #%    x   1-d signal
#    #%  Outputs
#    #%    y   filtered result
#    #%
#    #%  Description
#    #%    Filtering by periodic convolution of x with f
#    #%
#    #%  See Also
#    #%    aconv, UpDyadHi, UpDyadLo, DownDyadHi, DownDyadLo
#    #%
#    n = len(x);
#    p = len(f);
#    if p <= n:
#        xpadded = np.append(x[(n-p):n], x];
#    else:
#        z = np.zeros((1,p));
#
#        for i=xrange(p):
#            imod = (p*n -p + i-1) % n
#            z[i] = x[imod];
#
#        xpadded = np.append(z, x);
#
#    ypadded = filter(f,1,xpadded);
#    y = ypadded((p+1):(n+p));

def is_power2(num):
    return num != 0 and ((num & (num - 1)) == 0)
