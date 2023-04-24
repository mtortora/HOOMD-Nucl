##
##  NuclModifier.py
##
##  Created by mtortora on 12/12/2019.
##  Copyright © 2019 ENS Lyon. All rights reserved.
##

import ovito
import numpy as np

from matplotlib.cm import get_cmap
from ovito.data import Bonds, ParticleProperty


# ========================
# Particle parameters
# ========================

# Polymer color scheme - set to either "local", "global", "uniform" or "contour"
mode    = "contour"

# Auto-scale membrane bead size
scale   = True

# Bead size scaling factor (in fraction of average bond length)
factor  = 0.25

# Membrane stretching display range (in fraction of average bond length)
d_min   = -0.15
d_max   =  0.15

# Steps for bond normal calculations (for mode="global")
delta   = 25

# Discretisation parameters for non-uniform orientational order (for mode="local")
n_r     = 1
n_phi   = 10
n_theta = 5

# Colormap for order parameters
cmap_memb = get_cmap("viridis")
cmap_poly = get_cmap("RdYlBu_r")


# ========================
# Custom pipeline modifier
# ========================

def modify(frame, input, output):
	
	# Properties to be recomputed at every frame
	color_prop  = output.create_particle_property(ParticleProperty.Type.Color)
	radius_prop = output.create_particle_property(ParticleProperty.Type.Radius)
	
	# Load input variables
	size        = output.number_of_particles
	pos         = output.particle_properties.position.array
	
	ptypes      = output.particle_properties.particle_type.array
	btypes      = output.bond_properties.bond_type.array
	
	bonds       = output.bonds.array
	bonds_enum  = Bonds.Enumerator(output.bonds)
	
	# Number of membrane vertices and polymer beads
	n_vert      = np.count_nonzero(ptypes == 0)
	n_poly      = np.count_nonzero(ptypes > 0)
	
	# Number of harmonic springs and FENEs
	n_harm      = np.count_nonzero(btypes == 0)//2
	n_fene      = np.count_nonzero(btypes == 1)//2
	
	# Number of chains and beads per chain
	n_chain     = n_poly-n_fene
	n_bead      = n_poly//n_chain
	
	# Compute bond axes/lengths
	bonds_vec   = pos[bonds[:,1]] - pos[bonds[:,0]]
	bonds_vec  += np.dot(output.cell.matrix[:,:3], output.bonds.pbc_vectors.T).T
	
	bonds_lgt   = np.linalg.norm(bonds_vec, axis=-1)
	
	# Compute local membrane strains
	disps       = np.zeros(n_vert)
	count       = np.zeros(n_vert)
	
	for i in range(n_vert):
		for j in bonds_enum.bonds_of_particle(i):
			disps[i] += bonds_lgt[j]
			count[i] += 1.

	# Normalise strains in the range [d_min,d_max]
	disps  /= count
	diam    = disps.mean()
	
	strains = (disps-diam)/diam
	strains = (strains-d_min)/(d_max-d_min)
	
	# Color beads by local bond orientation
	if mode == "local":
		b           = bonds[::2,0][n_harm:n_harm+n_fene]
		
		t_vecs      = bonds_vec[::2][n_harm:n_harm+n_fene]
		t_lgts      = bonds_lgt[::2][n_harm:n_harm+n_fene]
		
		t_vecs     /= t_lgts[:,None]
		
		pos_bead    = pos[b,:].copy()
		pos_bead   -= pos_bead.mean(axis=0, keepdims=True)
		
		r2          = pos_bead[:,0]**2+pos_bead[:,1]**2
		r           = np.sqrt(r2+pos_bead[:,2]**2)
		
		phi         = np.arctan2(pos_bead[:,1], pos_bead[:,0])
		theta       = np.arctan2(np.sqrt(r2), pos_bead[:,2])
		
		r_max       = r.max() + 1e-4
		n_max       = n_r*n_phi*n_theta
		
		r_bins      = np.linspace(0, r_max**3, num=n_r+1)**(1/3.)
		p_bins      = np.linspace(-np.pi, np.pi, num=n_phi+1)
		t_bins      = np.arccos(2*np.linspace(1, 0, num=n_theta+1)-1)
		
		ids_r       = np.digitize(r, r_bins)-1
		ids_p       = np.digitize(phi, p_bins)-1
		ids_t       = np.digitize(theta, t_bins)-1
		
		ids         = ids_r*n_theta*n_phi+ids_p*n_theta+ids_t
		u0,u1,u2    = t_vecs.T
		
		# Compute molecular order-parameter tensor qs
		qs          = np.zeros((n_max,3,3))
		
		qs[:,0,0]  += np.bincount(ids, weights=1.5*u0*u0-0.5, minlength=n_max)
		qs[:,1,1]  += np.bincount(ids, weights=1.5*u1*u1-0.5, minlength=n_max)
		qs[:,2,2]  += np.bincount(ids, weights=1.5*u2*u2-0.5, minlength=n_max)
		
		qs[:,1,0]  += np.bincount(ids, weights=1.5*u1*u0, minlength=n_max)
		qs[:,2,0]  += np.bincount(ids, weights=1.5*u2*u0, minlength=n_max)
		qs[:,2,1]  += np.bincount(ids, weights=1.5*u2*u1, minlength=n_max)
		
		count       = np.bincount(ids, minlength=n_max)
		
		mask        = np.where(count, 1, 0)
		count       = np.where(count, count, 1.)
		
		# Spectral analysis of the locally-averaged tensor q
		qs         /= count[:,None,None]
		evals,evecs = np.linalg.eigh(qs)
		
		# Local order parameter (s)/director (dir) are the largest eval/evec of q
		ss          = evals[:,-1]
		dirs        = evecs[:,:,-1]
		
		rs          = (r_bins[1:]+r_bins[:-1])/2.
		ts          = (t_bins[1:]+t_bins[:-1])/2.
		
		drs         = np.diff(r_bins)
		dts         = np.diff(t_bins)
		
		rs          = np.repeat(rs,  n_phi*n_theta)
		drs         = np.repeat(drs, n_phi*n_theta)
		
		ts          = np.tile(ts,  n_phi*n_r)
		dts         = np.tile(dts, n_phi*n_r)
		
		dV          = rs**2 * np.sin(ts) * drs*dts * 2*np.pi/float(n_phi)
		s_mean      = (ss*dV*mask).sum()/(dV*mask).sum()
		
		# Transform bond indices b into contiguous particle indices p
		p           = b-n_vert-np.repeat(np.arange(n_chain), n_bead-1)
		
		# ops is the projection of local bond axes on dir
		ops         = ss[ids]*np.abs(u0[p]*dirs[ids,0]+u1[p]*dirs[ids,1]+u2[p]*dirs[ids,2])
		ops         = ops.reshape((n_chain,n_bead-1))
		
		ops         = np.insert(ops, 0, ops[:,0], axis=1)
		ops         = ops.flatten()
		
		print("Average OP: %.3f" % s_mean)

	# Color beads by global orientation of bond normal planes along the chains
	elif mode == "global":
		t_vecs = bonds_vec[::2][n_harm:n_harm+n_fene]
		t_vecs = t_vecs.reshape((n_chain,n_bead-1,3))
		
		n_vecs = np.cross(t_vecs[:,:-delta], t_vecs[:,delta:])
		n_lgts = np.linalg.norm(n_vecs, axis=-1, keepdims=True)
		
		if np.any(n_lgts < 1e-4):
			s   = 0.
			ops = np.zeros(n_poly)
		
		else:
			n_vecs     /= n_lgts
			qs          = np.zeros((n_vecs.shape[0],n_vecs.shape[1],3,3))
			
			u0          = n_vecs[...,0]
			u1          = n_vecs[...,1]
			u2          = n_vecs[...,2]
			
			qs[...,0,0] = u0*u0
			qs[...,1,1] = u1*u1
			qs[...,2,2] = u2*u2
			
			qs[...,1,0] = u1*u0
			qs[...,2,0] = u2*u0
			qs[...,2,1] = u2*u1
			
			qs          = 3/2. * qs.mean(axis=(0,1)) - 1/2. * np.eye(3)
			evals,evecs = np.linalg.eigh(qs)
			
			s           = evals[-1]
			dir         = evecs[:,-1]
			
			ops         = s*np.abs(dir[0]*u0 + dir[1]*u1 + dir[2]*u2)
			
			for i in range(delta//2+1):
				ops = np.insert(ops, ops.shape[1], ops[:,-1], axis=1)
				ops = np.insert(ops, 0, ops[:,0], axis=1)
			
			if delta % 2 == 0: ops = ops[:,:-1]
			
			ops = ops.flatten()
		
		print("Average OP: %.3f" % s)

	# Color individual chromosomes uniformly
	elif mode == "uniform":
		ops = np.arange(n_chain)/float(n_chain)
		ops = np.repeat(ops, n_bead)
	
	# Color beads by curvilinear abscissa
	elif mode == "contour": ops = np.arange(n_poly)/float(n_poly)
	
	# Color particles/membrane according to ops/strains
	colors                = np.zeros((size,3))
	
	colors[n_vert:,:]     = cmap_poly(ops)    [:,:3]
	colors[:n_vert,:]     = cmap_memb(strains)[:,:3]
	
	# Set membrane vertex diameter to mean bond distance if required
	radii                 = np.ones(size) * 0.5
	radii[:n_vert]        = 0.5*diam*factor if scale else 0.5
	
	color_prop.marray [:] = colors
	radius_prop.marray[:] = radii
