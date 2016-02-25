#!/usr/local/bin/python
import numpy as np
import pywt, math
import scipy as sp

wl = pywt.Wavelet('haar')
levels = 8

def denoise_waveform(wf_array, flatTimeSamples):
  #should already by a numpy array
  threshold_list = get_threshold_list()

  swt_output = pywt.swt(wf_array, wl, level=levels)
  # threshold the SWT coefficients
  apply_threshold(swt_output, 1., threshold_list)
  # inverse transform
  cA_thresh = iswt(swt_output, wl)
  
  wf_array = cA_thresh

  #re-baseline-subtract
  wf_array -= np.mean(wf_array[:flatTimeSamples])

  return wf_array



def get_threshold_list():
 return [ 206.2,
          575.1,
          447.7,
          313.6,
          229.1,
          207.2,
          524.8,
          780.2]

def iswt(coefficients, wavelet):
    """
      Input parameters: 

        coefficients
          approx and detail coefficients, arranged in level value 
          exactly as output from swt:
          e.g. [(cA1, cD1), (cA2, cD2), ..., (cAn, cDn)]

        wavelet
          Either the name of a wavelet or a Wavelet object

    """
    output = coefficients[0][0].copy() # Avoid modification of input data

    #num_levels, equivalent to the decomposition level, n
    num_levels = len(coefficients)
    for j in range(num_levels,0,-1): 
        step_size = int(math.pow(2, j-1))
        last_index = step_size
        _, cD = coefficients[num_levels - j]
        for first in range(last_index): # 0 to last_index - 1

            # Getting the indices that we will transform 
            indices = np.arange(first, len(cD), step_size)

            # select the even indices
            even_indices = indices[0::2] 
            # select the odd indices
            odd_indices = indices[1::2] 

            # perform the inverse dwt on the selected indices,
            # making sure to use periodic boundary conditions
            x1 = pywt.idwt(output[even_indices], cD[even_indices], wavelet, 'per') 
            x2 = pywt.idwt(output[odd_indices], cD[odd_indices], wavelet, 'per') 

            # perform a circular shift right
            x2 = np.roll(x2, 1)

            # average and insert into the correct indices
            output[indices] = (x1 + x2)/2.  

    return output
    
def apply_threshold(output, scaler = 1., input=None):
   """ 
       output is a list of vectors (cA and cD, approximation
       and detail coefficients) exactly as you would expect
       from swt decomposition.  
          e.g. [(cA1, cD1), (cA2, cD2), ..., (cAn, cDn)]

       If input is none, this function will calculate the
       tresholds automatically for each waveform.
       Otherwise it will use the tresholds passed in, assuming
       that the length of the input is the same as the length
       of the output list.
       input looks like:
          [threshold1, threshold2, ..., thresholdn]

       scaler is a tuning parameter that will be multiplied on
       all thresholds.  Default = 1 (0.8?)
   """
      
   for j in range(len(output)):
      cA, cD = output[j]
      if input is None:
        dev = np.median(np.abs(cD - np.median(cD)))/0.6745
        thresh = math.sqrt(2*math.log(len(cD)))*dev*scaler
      else: thresh = scaler*input[j]
      cD = pywt.threshold(cD, thresh, mode='hard')
      output[j] = (cA, cD)

if __name__=="__main__":
    main(sys.argv[1:])


