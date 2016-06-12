#!/usr/bin/env python

from numpy import *
import numpy as np
import sys
import os

BLOCK_SIZE = 16
hostfile_name = "/home/ubuntu/fangling/petuum/bosen/machinefiles/serverlist"

def generateData(sampleNumber,paraNumber):
	
	s = np.random.uniform(-2,2,paraOfNumber)
	s = mat(s).T
	np.savetxt("../data/solution.dat",s)

	for i in range(BLOCK_SIZE):
		Astr = "../data/A%d.dat"%i;bstr = "../data/b%d.dat"%i
		a = np.random.uniform(-1,1,paraOfNumber * sampleOfNumber)
		a = a.reshape((sampleOfNumber,paraOfNumber))
		np.savetxt(Astr,a)
		b = a * s
		print "the %d block data"%i
		np.savetxt(bstr,b)
		
if __name__ == "__main__":
	if len(sys.argv) < 3:
		print "argv must two praraments likes (samepleNubmer,paraNumber)!"
	sampleOfNumber = int(sys.argv[1])
	paraOfNumber = int(sys.argv[2])
	generateData(sampleOfNumber,paraOfNumber)

	fp = open(hostfile_name)
	for line in fp.readlines():
		line = line.strip().split()
		cmd = "scp -r ../data ubuntu@"+line[1]+":/home/ubuntu/fangling/petuum/bosen/app/ADMMforNorm/"
		cmd1 = "scp -r "+hostfile_name+" ubuntu@"+line[1]+":/home/ubuntu/fangling/petuum/bosen/machinefiles"
		print cmd
		os.system(cmd)
		os.system(cmd1)
