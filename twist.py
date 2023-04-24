##
##  twist.py
##
##  Created by mtortora on 12/12/2019.
##  Copyright © 2019 ENS Lyon. All rights reserved.
##

import os
import sys

import gsd.hoomd
import numpy as np


# Input/output
if len(sys.argv) != 3:
	print("\033[1;31mUsage is %s trajectory turns\033[0m" % sys.argv[0])
	sys.exit()
	
path_traj = os.path.realpath(sys.argv[1])

dir_traj  = os.path.dirname(path_traj)
path_twist = os.path.join(dir_traj, 'twisted_%s.gsd' % sys.argv[2])

traj = gsd.hoomd.open(path_traj, 'rb')
twist = float(sys.argv[2]) * 2*np.pi


def rotation_matrix(axis, theta):
	axis /= np.linalg.norm(axis)
	
	a = np.cos(theta/2.)
	b, c, d = -axis*np.sin(theta/2.)
	
	aa, bb, cc, dd = a*a, b*b, c*c, d*d
	bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
	
	return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
					 [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
					 [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])
					 

frame = traj[-1]

pos = frame.particles.position.copy()
pos -= pos.mean(axis=0, keepdims=1)

_, d, v = np.linalg.svd(pos, full_matrices=False)
axis = v[0]

proj = (pos*axis[None,:]).sum(axis=1)

pmin = proj.min()
pmax = proj.max()

for i in range(len(pos)):
	ang = (proj[i]-pmin)/(pmax-pmin)*twist
	pos[i] = np.dot(rotation_matrix(axis, ang), pos[i])
	
frame.particles.position = pos

output = gsd.hoomd.open(path_twist, 'wb')
output.append(frame)

print("\033[1;32mPrinted twisted configuration to '%s'\033[0m" % path_twist)
