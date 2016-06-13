#!/usr/bin/env python

import os
from os.path import dirname, join

#host file
hostfile_name = "machinefiles/serverlist"

app_dir = dirname(dirname(os.path.realpath(__file__)))
proj_dir = dirname(dirname(app_dir))
hostfile = join(proj_dir, hostfile_name)

#kill program
killcmd = app_dir+"/script/kill.py "+hostfile
os.system(killcmd)

fp = open(hostfile)

for ip in fp.readlines():
	ip = ip.strip().split()
	if ip[1] != "10.10.10.63":
		cmd_cp = "scp "+join(app_dir,"script/")+"run_local.py " + "ubuntu@%s:"%(ip[1])
		cmd_cp += join(app_dir,"script/")
	
		cmd_cp1 = "scp "+join(app_dir,"bin/")+"linearRegression_main " + "ubuntu@%s:"%(ip[1])
		cmd_cp1 += join(app_dir,"bin")
		
		print cmd_cp
		print cmd_cp1
		
		os.system(cmd_cp)
		os.system(cmd_cp1)
fp.close()

