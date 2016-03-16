import numpy as np
import matplotlib.pyplot as plt


def plotWPFile(fileName, pcDepth, pcRadius, taperLength):
  f = plt.figure()
  (r, z, wp) = np.loadtxt(fileName, unpack=True)
  nr = np.amax(r)*10+1
  nz =  np.amax(z)*10+1
  grid = wp.reshape(nr,nz)
  grid[0:pcRadius*10,0:pcDepth*10] = np.nan
  
  #hmm, what about the 45 degree taper?
  
  for iZ in np.arange(0, taperLength, 0.1):
    zHeight = taperLength-iZ
    taper_max_r_idx = 10*np.around(np.amax(r) - zHeight*np.tan(np.pi/4), decimals=1)
#    print "z: %f, r: %f" % (zHeight, taper_max_r)
    grid[taper_max_r_idx:, iZ*10] = np.nan
  
  
  grid = grid.T
  
  grid_neg = np.fliplr(grid)
  
  grid_full = np.hstack((grid_neg,grid))
  
  #plt.imshow(grid[:50,:50], origin='lower', extent=(0,5,0,5), interpolation='nearest', cmap=plt.cm.Paired)
  plt.title(fileName)
  plt.imshow(grid_full, origin='lower', extent=(-1*r.max(), r.max(), z.min(), z.max()), interpolation='nearest', cmap=plt.cm.RdYlBu_r)
  return wp

def compareWPFile(fileName1, fileName2):
  (r, z, wp1) = np.loadtxt(fileName1, unpack=True)
  (r, z, wp2) = np.loadtxt(fileName2, unpack=True)

  for (i, wp1_val) in enumerate(wp1):
    diff = wp1_val - wp2[i]
    if diff !=0:
      print "difference at (r=%f,z=%f)" % (r[i],z[i])

#def compareWPFile(fileName1, fileName2, pcDepth, pcRadius, taperLength):
#  f = plt.figure()
#  (r, z, wp1) = np.loadtxt(fileName1, unpack=True)
#  nr = np.amax(r)*10+1
#  nz =  np.amax(z)*10+1
#  grid1 = wp1.reshape(nr,nz)
#  
#  (r, z, wp2) = np.loadtxt(fileName2, unpack=True)
#  nr = np.amax(r)*10+1
#  nz =  np.amax(z)*10+1
#  grid2 = wp2.reshape(nr,nz)
#  
#  grid = grid1 - grid2
#  
#  grid[0:pcRadius*10,0:pcDepth*10] = np.nan
#  
#  #hmm, what about the 45 degree taper?
#  
#  for iZ in np.arange(0, taperLength, 0.1):
#    zHeight = taperLength-iZ
#    taper_max_r_idx = 10*np.around(np.amax(r) - zHeight*np.tan(np.pi/4), decimals=1)
##    print "z: %f, r: %f" % (zHeight, taper_max_r)
#    grid[taper_max_r_idx:, iZ*10] = np.nan
#  
#  
#  grid = grid.T
#  
#  
#  
#  #plt.imshow(grid[:50,:50], origin='lower', extent=(0,5,0,5), interpolation='nearest', cmap=plt.cm.Paired)
#  plt.title("%s minus %s" %(fileName1, fileName2))
#  plt.imshow(grid, origin='lower', extent=(r.min(), r.max(), z.min(), z.max()), interpolation='nearest', cmap=plt.cm.flag)



plotWPFile("P42661C_bulletized_wp.dat",1.8, 1.5, 4.5)

#plotWPFile("P42661C_wp.dat",1.8, 1.5, 4.5)

compareWPFile("P42661C_bulletized_wp.dat","P42661C_wp.dat")#,1.8, 1.5, 4.5)

#f = plt.figure(2)
#plotWPFile("P42712A_wp.dat", 1.9, 1.6, 4.5)


plt.show()


