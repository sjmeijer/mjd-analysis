import numpy as np
import matplotlib.pyplot as plt
import os, array, sys


def main(argv):
  for i in os.listdir(os.getcwd()):
  
      energy = 100
      spParam = 100
  
      if i.endswith(".pdf"):
          split = i.split("_")
          for j in split:
            if j.startswith("energy"):
              energy = float(j.split("energy")[-1])
            if j.startswith("spparam"):
              #pull off the .pdf
              k = j.split(".")[0]
              spParam = float(k.split("spparam")[-1])
  
          if energy < 1 and spParam < 10:
            os.rename(i, "interesting/" + i)
      
      else:
          continue

if __name__=="__main__":
    main(sys.argv[1:])
