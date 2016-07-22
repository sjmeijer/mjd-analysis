#!/usr/local/bin/python
import sys, os, glob

def main(argv):

  old_string = "conf/files/"

  for filename in glob.glob('conf/*.conf'):
    print "Opening file %s" % filename
    
    doChange = 1
    with open(filename) as f:
      s = f.read()
      if old_string not in s:
          print '"{old_string}" not found in {filename}.'.format(**locals())
          doChange = 0
          
    if doChange:
      with open(filename, "w") as f:
        s = s.replace(old_string, "conf/fields/")
        #print s
        f.write(s)



if __name__=="__main__":
    main(sys.argv[1:])


