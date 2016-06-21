#!/usr/bin/env python

from numpy import *
from os.path import join
from os.path import dirname
import numpy as np
import sys
import os

#TYPE = "dense"
TYPE = "sparse"

BLOCK_SIZE = 64
hostfile_name = "/home/ubuntu/fangling/petuum/bosen/machinefiles/serverlist"

app_dir = dirname(dirname(os.path.realpath(__file__)))
data_dir = join(app_dir,"data")

if not os.path.exists(data_dir):
	os.makedirs(data_dir)

if len(sys.argv) < 3:
	print "argv must two praraments likes (samepleNubmer,paraNumber)!"
	exit()
	
sampleOfNumber = int(sys.argv[1])
paraOfNumber = int(sys.argv[2])

s = np.random.uniform(-2,2,paraOfNumber)
if TYPE == "sparse":
	zeroloc = np.random.randint(paraOfNumber,size = paraOfNumber/2)
	for i in zeroloc:
		s[i] = 0
s = mat(s).T
np.savetxt(data_dir+"/solution.dat",s)

for i in range(BLOCK_SIZE):
	Astr = data_dir+"/A%d.dat"%i;bstr = data_dir+"/b%d.dat"%i
	a = np.random.uniform(-1,1,paraOfNumber * sampleOfNumber)
	a = a.reshape((sampleOfNumber,paraOfNumber))
	np.savetxt(Astr,a)
	b = a * s
	print "the %d block data"%i
	np.savetxt(bstr,b)


fp = open(hostfile_name)
for line in fp.readlines():
	line = line.strip().split()
	cmd = "scp -r "+ data_dir +" ubuntu@"+line[1]+":"+app_dir
	cmd1 = "scp "+hostfile_name+" ubuntu@"+line[1]+":/home/ubuntu/fangling/petuum/bosen/machinefiles"
	print cmd
	os.system(cmd)
	os.system(cmd1)
