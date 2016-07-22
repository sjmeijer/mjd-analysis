#!/usr/local/bin/python
import sys, os, glob

def main(argv):

  old_string = "conf/files/"

  for filename in glob.glob('conf/*.conf'):
    print "Opening file %s" % filename
    
    with open(filename, "a") as f:

      f.write("bulletize_PC    1")



if __name__=="__main__":
    main(sys.argv[1:])


