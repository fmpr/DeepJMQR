import sys
import numpy as np

f = open(sys.argv[1])
header = f.readline()
print header.strip()
vals = [[] for x in header.split(",")]
for line in f:
	splt = line.strip().split(",")
	for i in xrange(len(splt)):
		vals[i].append(float(splt[i]))
f.close()

s = ""
for col in vals:
	s += "%.3f," % (np.mean(col),)
print "MEANS:", s

s = ""
for col in vals:
	s += "%.3f," % (np.std(col),)
print "STDDEVS:", s
