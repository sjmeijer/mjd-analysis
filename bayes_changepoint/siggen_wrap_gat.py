#!/usr/local/bin/python
from ROOT import *
TROOT.gApplication.ExecuteFile("$MGDODIR/Root/LoadMGDOClasses.C")
TROOT.gApplication.ExecuteFile("$MGDODIR/Majorana/LoadMGDOMJClasses.C")


import matplotlib.pyplot as plt

confFile = "P42661C_autogen_final.conf"

#requires a PSS implementation on $GATDIR (obvs)
siggenInst = GATSiggenInstance(confFile)


weightedClusterPosition = TVector3(0, 15, 10);

clusterWaveform = MGTWaveform();
calcFlag = siggenInst.CalculateWaveform(weightedClusterPosition, clusterWaveform, 5);

np_data = clusterWaveform.GetVectorData()

fig1 = plt.figure(1)

plt.plot( np_data  ,color="red" )

plt.show()
