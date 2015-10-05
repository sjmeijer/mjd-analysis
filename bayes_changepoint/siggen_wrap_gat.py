#!/usr/local/bin/python
from ROOT import *
TROOT.gApplication.ExecuteFile("$MGDODIR/Root/LoadMGDOClasses.C")
TROOT.gApplication.ExecuteFile("$MGDODIR/Majorana/LoadMGDOMJClasses.C")


import matplotlib.pyplot as plt

confFile = "P42661C_autogen_final.conf"

#requires a PSS implementation on $GATDIR (obvs)
siggenInst = GATSiggenInstance(confFile)

#x,y,x position (I don't have it coded in yet to switch to R, phi, z but I can do that pretty easy if necessary)
weightedClusterPosition = TVector3(0, 15, 10);

clusterWaveform = MGTWaveform(); #gets passed as a pointer in the c++ code
calcFlag = siggenInst.CalculateWaveform(weightedClusterPosition, clusterWaveform, 5);

#the siggen waveform will be have whatever sampling period and length is specified in the conf file.  Keep it at 10 ns so its the same as gretina data.

np_data = clusterWaveform.GetVectorData()

fig1 = plt.figure(1)
plt.plot( np_data  ,color="red" )
plt.show()
