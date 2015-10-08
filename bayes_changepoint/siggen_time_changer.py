#!/usr/local/bin/python


baseDriftVelFileName = "dat/drift_vel_tcorr.tab"
multiplier = 0.9


between_cols = "      "

def main():

    outFileName = baseDriftVelFileName.split(".")[0] + "_edited.tab"
    baseFile = open(baseDriftVelFileName, 'r')
    
    outFileName = open(outFileName, 'w')
    
    for line in baseFile:
        if str.startswith(line, "#") or str.startswith(line, "e") or str.startswith(line, "h"):
            outFileName.write(line)
            continue
        cols = line.split()
        if len(cols) == 0:
            outFileName.write("\n")
        else:
            newline = between_cols + cols[0]
            for i in range(1, len(cols)):
                newline += between_cols
                newline += "%0.4f" % (multiplier * float(cols[i]))
            outFileName.write(newline + "\n")

    baseFile.close()



if __name__=="__main__":
    main()