import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.style.use('presentation')

import matplotlib
#print matplotlib.get_configdir()
#exit()

def plotWPFile(fileName, pcDepth, pcRadius, taperLength):
  f = plt.figure(figsize=(10,5))
  (r, z, c, e, er, ez ) = np.loadtxt(fileName, unpack=True)
  nr = np.int(np.amax(r)*10+1)
  nz =   np.int(np.amax(z)*10+1)
  
  grid = e.reshape(nr,nz)
#  grid[0:pcRadius*10,0:pcDepth*10] = np.nan

  #hmm, what about the 45 degree taper?
  
  for iZ in np.arange(0, taperLength, 0.1):
    zHeight = taperLength-iZ
    taper_max_r_idx = np.int(10*np.around(np.amax(r) - zHeight*np.tan(np.pi/4), decimals=1))
#    print "z: %f, r: %f" % (zHeight, taper_max_r)
    grid[taper_max_r_idx:, np.int(iZ*10)] = np.nan


  grid[np.where(grid == 1)] = np.nan

  grid = grid.T
  
  print "shape is " + str( grid.shape )
  
  grid_neg = np.fliplr(grid)
  
  grid_full = np.hstack((grid_neg,grid))
  
  #plt.imshow(grid[:50,:50], origin='lower', extent=(0,5,0,5), interpolation='nearest', cmap=plt.cm.Paired)
  im = plt.imshow(grid_full, origin='lower', extent=(-1*r.max(), r.max(), z.min(), z.max()), interpolation='nearest', cmap='afmhot',  )
  
  
  plt.colorbar()
  
#  ax = plt.gca()
#  divider = make_axes_locatable(ax)
#  cax = divider.append_axes("right", size="5%", pad=0.05)
#  
#  plt.colorbar(im, cax=cax)

  plt.xlabel("Radius (mm)")
  plt.ylabel("Height (mm)")
  plt.tight_layout()
  plt.savefig("efield.pdf")
  
#  plt.xlim(0,5)
#  plt.ylim(0,5)




plotWPFile("P42661C_ev.dat",1.8, 1.5, 4.5)

#plotWPFile("P42661C_wp.dat",1.8, 1.5, 4.5)

#compareWPFile("P42661C_wp.dat","P42661C_wp.dat")#,1.8, 1.5, 4.5)

#f = plt.figure(2)
#plotWPFile("P42712A_wp.dat", 1.9, 1.6, 4.5)


plt.show()


