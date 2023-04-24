##
##  HoomdNucl.py
##
##  Created by mtortora on 12/12/2019.
##  Copyright © 2019 ENS Lyon. All rights reserved.
##

import os
import sys
import hoomd
import GPUtil
import codecs
import argparse

import numpy as np

from hoomd import md
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist


# ========================
# ANSI escape sequences
# ========================

ANSIColor = {
    "bla": "1;30m",
    "red": "1;31m",
    "gre": "1;32m",
    "yel": "1;33m",
    "blu": "1;34m",
    "pur": "1;35m",
    "cya": "1;36m",
    "whi": "1;37m"
}


# ========================
# Parse input parameters
# ========================

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("n_beads", help='number of beads per chromosome',
                    type=int, metavar='Nb')
parser.add_argument("n_chrom", help='number of chromosomes',
                    type=int, metavar='Nc')
parser.add_argument("eta", help='relaxed chromatin volume fraction',
                    type=float, metavar='eta')
parser.add_argument("--lp_soft", help='chromosome (soft) persistence length',
                    type=float, default=0.)
parser.add_argument("--lp_stiff", help='chromosome (stiff) persistence length',
                    type=float, default=50.)
parser.add_argument("--rm_wca", help='minimum polymer-membrane repulsion radius',
                    type=float, default=1.)
parser.add_argument("--r_memb", help='initial membrane radius (in chromatin bead diameter)',
                    type=float, default=60.)
parser.add_argument("--mode_init", help='initial chromosome conformation',
                    type=str, default="unknotted", choices=set(("unknotted", "brownian", "dense")))
parser.add_argument("--k", help='membrane Young modulus (in kT/bead_diameter^2)',
                    type=float, default=2500)
parser.add_argument("--r0", help='membrane spring rest length (in chromatin bead diameter)',
                    type=float, default=1)
parser.add_argument("--k_stretch", help='chromatin stretching rigidity (in kT/bead_diameter^2)',
                    type=float, default=15.)
parser.add_argument("--D0", help='LJ potential depth (in kT)',
                    type=float, default=0.)
parser.add_argument("--rc_at", help='LJ potential cutoff (in bead diameters)',
                    type=float, default=0)
parser.add_argument("--k_dihed", help='membrane bending rigidity (in kT)',
                    type=float, default=15.)
parser.add_argument("--sweep", help='run sequential batch of simulations over range of chosen parameters '
                                    '(choose any subset of "D","k","lp","r","force")',
                    type=str, default=None)
parser.add_argument("--nlist", help='type of neighbour list to be used',
                    type=str, default="tree", choices=set(("cell", "tree")))
parser.add_argument("--file_init", help='specify custom initial configuration file',
                    type=str, default=None)
parser.add_argument("--frame_init", help='specify initial frame index in file_init',
                    type=int, default=0)
parser.add_argument("--propagate", help='stiffness propagation run',
                    dest="propagate", action="store_true")
parser.add_argument("--decondense", help='decondensation run',
                    dest="decondense", action="store_true")
parser.add_argument("--restart", help='force simulation to restart from relaxed intial configuration',
                    dest="restart", action="store_false")
parser.add_argument("--hoomd_opt", help='additional HooMD flags',
                    default=[], nargs=argparse.REMAINDER)


# ========================
# Simulation manager
# ========================

class HoomdSim():

    def __init__(self,
                 n_beads, n_chrom, eta, lp_soft, lp_stiff,
                 rm_wca, r_memb,
                 mode_init,
                 k, r0, k_stretch,
                 D0, rc_at, k_dihed,
                 sweep, nlist,
                 file_init, frame_init, propagate, decondense, restart,
                 hoomd_opt):
        """Setup simulation"""

        # Set file paths
        path_top = "nb%d_nc%d_eta%.2f" % (n_beads, n_chrom, eta)

        if sweep:
            sweep = sweep.split()

            if any(x not in ["D", "k", "lp_soft", "lp_stiff", "r", "force"] for x in sweep):
                print("Choose sweep parameters from 'D','k','lp_soft','lp_stiff','r','force'")
                sys.exit()

            sweep_str = "".join(sorted(sweep))
            path_sim = os.path.join(path_top, "%ssw" % sweep_str)

            if "k" not in sweep: path_sim += "_k%.2f" % k
            if "r" not in sweep: path_sim += "_r%.2f" % r_memb
            if "lp_soft" not in sweep: path_sim += "_lpsoft%.0f" % lp_soft
            if "lp_stiff" not in sweep: path_sim += "_lpstiff%.0f" % lp_stiff

            if "D" in sweep:
                if rc_at == 0:
                    print("Need positive cutoff radius for LJ sweeping run -- aborting")
                    sys.exit()

                else:
                    path_sim += "_rc%.1f" % (rc_at)

        else:
            path_sim = os.path.join(path_top, "r%.2f_k%.2f_lpsoft%.0f_lpstiff%.0f" % (r_memb, k, lp_soft, lp_stiff))

        if (rc_at > 0) & (D0 > 0):
            if not sweep or "D" not in sweep: path_sim += "_D%.2f_rc%.1f" % (D0, rc_at)

        self.path_rest = os.path.join(path_sim, "restart.gsd")
        self.path_init = self.path_rest if not file_init else file_init

        self.path_rel = os.path.join(path_top, "relaxed_%s_k%.2f_r%.2f.gsd" % (mode_init, k, r_memb))
        self.path_simp = os.path.join(path_top, "simplices_%s_k%.2f_r%.2f.res" % (mode_init, k, r_memb))

        # Check existing files for run setup
        if (not os.path.exists(self.path_rel)) & (not os.path.exists(self.path_init)):
            self.relaxed = False
            self.restart = False

            if not os.path.exists(path_top): os.makedirs(path_top)

        else:
            self.relaxed = True
            self.restart = restart & (os.path.exists(self.path_init))

            if not os.path.exists(path_sim): os.makedirs(path_sim)

        if self.relaxed:
            path_log = os.path.join(path_sim, "log.out")

            self.path_ana = os.path.join(path_sim, "analysis.log")
            self.path_traj = os.path.join(path_sim, "trajectory.gsd")

        else:
            path_log = os.path.join(path_top, "log_%s_k%.2f_r%.2f.out" % (mode_init, k, r_memb))

            self.path_ana = os.path.join(path_top, "analysis_%s_k%.2f_r%.2f.log" % (mode_init, k, r_memb))
            self.path_traj = os.path.join(path_top, "trajectory_%s_k%.2f_r%.2f.gsd" % (mode_init, k, r_memb))

        self.overwrite = (not self.restart) | (file_init is not None)
        self.file_log = open(path_log, mode='w' if self.overwrite else 'a')

        # r_final is the membrane radius required to achieve the desired chromatin density
        r_final = ((n_beads + 1) * n_chrom * 3. / (16. * eta)) ** (1 / 3.)

        # n_vert is the number of membrane beads, assuming their dense sphere packing at r=r_final
        n_vert = int(8 * np.pi / np.sqrt(3) * r_final ** 2)

        self.n_tot = n_beads * n_chrom + n_vert

        self.n_beads = n_beads
        self.n_chrom = n_chrom
        self.n_vert = n_vert

        self.r_memb = r_memb
        self.rm_wca = rm_wca

        self.r0 = r_memb / r_final if r0 == 0 else r0
        self.k = k

        self.k_stretch = k_stretch

        self.lp_soft = lp_soft
        self.lp_stiff = lp_stiff

        self.k_dihed = k_dihed

        self.D0 = D0
        self.rc_at = rc_at

        self.sweep = sweep
        self.nlist = nlist

        self.mode_init = mode_init
        self.frame_init = frame_init

        self.propagate = propagate
        self.decondense = decondense

        # Select GPU/CPU based on availability
        gpu_avail = GPUtil.getAvailable(order='last', maxLoad=0.1)
        hoomd_device = "--gpu=%d" % gpu_avail[-1] if gpu_avail else "--mode=cpu"

        # Initialise HooMD with optional extra flags
        hoomd.context.initialize(" ".join(hoomd_opt) if hoomd_opt else hoomd_device)


    def build(self):
        """Setup simulation box and force fields"""

        self.build_system()

        if not self.relaxed:
            self.set_relax_field()
        else:
            self.set_full_field()

        self.set_output()


    def build_system(self):
        """Setup particles and topology"""

        if not self.relaxed:
            self.log("Could not find relaxed initial configuration - running in relaxation mode", 'pur')

            chroms = self.build_chroms()
            snap = hoomd.data.make_snapshot(N=self.n_tot,
                                            box=hoomd.data.boxdim(L=3 * self.r_memb, dimensions=3),
                                            particle_types=['membrane', 'poly_soft', 'poly_stiff', 'poly_growth'])
            self.build_config(snap, chroms)

        elif not self.restart:
            self.log("Starting from relaxed configuration %s" % self.path_rel, 'pur')

            snap = hoomd.data.gsd_snapshot(filename=self.path_rel)
            snap.box = hoomd.data.boxdim(L=2.25 * self.r_memb, dimensions=3)

            self.build_topology(snap)

        else:
            self.log("Starting from initial configuration %s (frame %d)" % (self.path_init, self.frame_init), 'pur')

            snap = hoomd.data.gsd_snapshot(filename=self.path_init, frame=self.frame_init)

        for i in range(self.n_vert): snap.particles.diameter[i] = self.r0

        self.log('Initial membrane bond length: %.2f' % self.r0, 'pur')

        if self.sweep and ("force" in self.sweep):
            self.dir = np.random.random(3)
            self.dir /= np.linalg.norm(self.dir)

            pos = snap.particles.position
            pos -= pos.mean(axis=0, keepdims=True)
			
            pos_vert = pos[:self.n_vert]
            proj = (pos_vert*self.dir[None,:]).sum(axis=1)
		
            idmin = np.argmin(proj)
            idmax = np.argmax(proj)
		
            bonds = snap.bonds.group[snap.bonds.typeid==0]
            snap.box = hoomd.data.boxdim(L=7 * self.r_memb, dimensions=3)

            lmin = []
            lmax = []
		
            for b in bonds:
                if idmin in b: lmin.append(b[0] if idmin==b[1] else b[1])
                if idmax in b: lmax.append(b[0] if idmax==b[1] else b[1])

            self.system = hoomd.init.read_snapshot(snap)
        
            self.npole = hoomd.group.tag_list(name="north-pole", tags=[idmax])
            self.spole = hoomd.group.tag_list(name="south-pole", tags=[idmin])
        
            self.ngroup = hoomd.group.tag_list(name="north-bonds", tags=lmax)
            self.sgroup = hoomd.group.tag_list(name="south-bonds", tags=lmin)
        
        else:
            self.system = hoomd.init.read_snapshot(snap)


    def build_chroms(self, pad=0.05):
        """Build initial chromosome conformation"""

        if self.mode_init == "unknotted":
            chroms = self.build_unknotted_chroms(pad)
        elif self.mode_init == "brownian":
            chroms = self.build_brownian_chroms(pad)
        elif self.mode_init == "dense":
            chroms = self.build_dense_chroms(pad)

        return chroms


    def build_unknotted_chroms(self, pad, n_tries=100, c_max=2):
        """Build random unknotted chromosome conformation"""

        n_max = c_max * self.n_beads
        d_max = int(np.floor(self.r_memb - self.r0 - pad))

        pos_init = np.zeros(3)
        chroms = []

        # Diffuse plane by plane along the z axis
        while True:
            while len(chroms) < self.n_chrom:
                idx_jump = 0
                counter = 0

                # Set chromosome extremity randomly near the southern pole of the membrane surface
                if len(chroms) == 0:
                    pos_init = np.asarray([0., 0., -d_max])
                else:
                    theta = 0.5 * np.pi * (1. + np.random.random())
                    phi = 2 * np.pi * np.random.random()

                    pos_init[0] = d_max * np.sin(theta) * np.cos(phi)
                    pos_init[1] = d_max * np.sin(theta) * np.sin(phi)
                    pos_init[2] = (d_max * np.cos(theta)).astype(int)

                chrom = [pos_init]

                chroms_built = np.asarray(chroms).reshape((-1, 3))
                chroms_in_plane = chroms_built[chroms_built[:, 2] == -chrom[-1][2]]

                while len(chrom) < self.n_beads:
                    disp = np.zeros(3)
                    bead = chrom[-1].copy()

                    # beads_in_plane is the list of existing beads in current z-plane
                    beads_in_plane = list(chroms_in_plane) + chrom[idx_jump:]

                    for j in range(n_tries):
                        disp[:2] = np.random.random(2) - 0.5
                        disp /= np.linalg.norm(disp)

                        new_bead = bead + disp

                        norm = np.linalg.norm(new_bead)
                        min_dist = np.min(cdist([new_bead], beads_in_plane))

                        success = (min_dist > 1.) & (norm < d_max)

                        if success:
                            chrom.append(new_bead)
                            break

                    if not success:
                        for j in range(n_tries):
                            new_bead = chrom[-1].copy()
                            new_bead[2] += 1.

                            norm = np.linalg.norm(new_bead)

                            chroms_in_plane = chroms_built[chroms_built[:, 2] == new_bead[2]]

                            if len(chroms_in_plane) > 0:
                                min_dist = np.min(cdist([new_bead], chroms_in_plane))
                                success = (min_dist > 1.) & (norm < d_max)

                            else:
                                success = (norm < d_max)

                            if success:
                                idx_jump = len(chrom)
                                chrom.append(new_bead)

                                break

                            else:
                                del chrom[-1]

                                if len(chrom) <= idx_jump: idx_jump = 0

                    counter += 1

                    if counter % 1000 == 0:
                        n_placed = len(chrom) + len(chroms_built)

                        self.log("Placed %d out of %d beads" % (n_placed, self.n_beads * self.n_chrom), 'whi')

                    if counter > n_max:
                        self.log("Failed to grow full chain within chosen membrane radius - retrying", 'yel')
                        break

                if len(chrom) == self.n_beads:
                    chrom = np.asarray(chrom)
                    com = np.mean(chrom, axis=0)

                    chrom[:, 2] -= int(com[2])

                    while np.any(np.linalg.norm(chrom, axis=1) > d_max): chrom[:, 2] += np.sign(com[2])

                    chroms.append(chrom)

                    self.log("Successfully built %d out of %d chromosomes" % (len(chroms), self.n_chrom), 'blu')

            if len(chroms) == self.n_chrom:
                chroms = np.asarray(chroms)
                break

        return chroms


    def build_brownian_chroms(self, pad, n_tries=100, c_max=2):
        """Build brownian (potentially-knotted) chromosome conformation"""

        n_max = c_max * self.n_beads
        d_max = int(np.floor(self.r_memb - self.r0 - pad))

        pos_init = np.zeros(3)
        chroms = []

        # Diffuse plane by plane along the z axis
        while True:
            while len(chroms) < self.n_chrom:
                counter = 0

                # Set chromosome extremity randomly near the southern pole of the membrane surface
                if len(chroms) > 0:
                    theta = np.pi * (1. + np.random.random())
                    phi = 2 * np.pi * np.random.random()

                    pos_init[0] = d_max * np.sin(theta) * np.cos(phi)
                    pos_init[1] = d_max * np.sin(theta) * np.sin(phi)
                    pos_init[2] = (d_max * np.cos(theta)).astype(int)

                chrom = [pos_init]
                chroms_built = np.asarray(chroms).reshape((-1, 3))

                while len(chrom) < self.n_beads:
                    bead = chrom[-1].copy()

                    # beads_placed is the list of existing beads
                    beads_placed = np.append(chroms_built, [chrom], axis=0)

                    for j in range(n_tries):
                        disp = np.random.random(3) - 0.5
                        disp /= np.linalg.norm(disp)

                        new_bead = bead + disp

                        norm = np.linalg.norm(new_bead)
                        min_dist = np.min(cdist([new_bead], beads_placed))

                        success = (min_dist >= 1.) & (norm < d_max)

                        if success:
                            chrom.append(new_bead)
                            break

                    if not success: del chrom[-1]

                    counter += 1

                    if counter % 1000 == 0:
                        n_placed = len(chrom) + len(chroms_built)

                        self.log("Placed %d out of %d beads" % (n_placed, self.n_beads * self.n_chrom), 'whi')

                    if counter > n_max:
                        self.log("Failed to grow full chain within chosen membrane radius - retrying", 'yel')
                        break

                if len(chrom) == self.n_beads:
                    chroms.append(chrom)

                    self.log("Successfully built %d out of %d chromosomes" % (len(chroms), self.n_chrom), 'blu')

            if len(chroms) == self.n_chrom:
                chroms = np.asarray(chroms)
                break

        return chroms


    def build_dense_chroms(self, pad):
        """Build dense chromosome conformation"""

        # n_min is the length of the smallest cube containing the self-avoiding chromatin
        n_min = int(np.ceil((self.n_beads * self.n_chrom) ** (1 / 3.)))
        r_min = (n_min * np.sqrt(3) / 2 + pad) * 1. / (1. - self.r0 / self.r_memb)

        if r_min < self.r_memb:
            self.log("Minimum radius of %.3f is smaller than r_memb - using r_memb instead" % r_min, 'whi')

        else:
            self.log("Chosen membrane radius too small - pick value > %.3f" % r_min, 'red')
            sys.exit()

        x_up = True
        z_up = True

        chroms = [-n_min / 2. * np.ones(3)]

        while len(chroms) < self.n_beads * self.n_chrom:
            bead = chroms[-1].copy()
            new_bead = bead + np.array([0, 0, 1]) if z_up else bead - np.array([0, 0, 1])

            if np.abs(new_bead[2]) > n_min / 2:
                new_bead = bead + np.array([1, 0, 0]) if x_up else bead - np.array([1, 0, 0])
                z_up ^= True

                if np.abs(new_bead[0]) > n_min / 2:
                    new_bead = bead + np.array([0, 1, 0])
                    x_up ^= True

            chroms.append(new_bead)

        chroms = np.asarray(chroms)
        chroms -= np.mean(chroms, axis=0, keepdims=True)

        chroms = chroms.reshape((self.n_chrom, self.n_beads, 3))

        return chroms


    def build_config(self, snap, chroms):
        """Setup membrane configuration with random uniform vertex distribution"""

        vertices = np.zeros((self.n_vert, 3))

        thetas = np.arccos(2 * np.random.random(self.n_vert) - 1.)
        phis = 2 * np.pi * np.random.random(self.n_vert)

        vertices[:, 0] = self.r_memb * np.sin(thetas) * np.cos(phis)
        vertices[:, 1] = self.r_memb * np.sin(thetas) * np.sin(phis)
        vertices[:, 2] = self.r_memb * np.cos(thetas)

        for i in range(self.n_vert):
            snap.particles.position[i] = vertices[i]

            snap.particles.typeid[i] = 0
            snap.particles.charge[i] = 1.
            snap.particles.diameter[i] = self.r0

        for i in range(self.n_chrom):
            for j in range(self.n_beads):
                k = self.n_vert + i * self.n_beads + j

                snap.particles.position[k] = chroms[i, j]

                snap.particles.typeid[k] = 1
                snap.particles.charge[k] = 0.
                snap.particles.diameter[k] = 1.


    def build_topology(self, snap):
        """Build membrane/chromosome topology"""

        positions = snap.particles.position.copy()
        vertices = positions[:self.n_vert]

        snap.bonds.types = ['mb_bonds', 'pl_bonds']
        snap.angles.types = ['pl_soft', 'pl_stiff', 'pl_growth']
        snap.dihedrals.types = ['mb_diheds']

        # Delauney triangulation is obtained from the convex hull of membrane vertices
        hull = ConvexHull(vertices)
        n_simp = len(hull.simplices)

        mb_bonds = []
        mb_diheds = []

        for i in range(n_simp):
            simp = hull.simplices[i]
            nb = hull.neighbors[i]

            for j in range(3):
                if nb[j] > i:
                    edge = np.delete(simp, j)
                    simp_nb = hull.simplices[nb[j]]

                    k = set(simp_nb) - set(edge)
                    dihed = [simp[j]] + list(edge) + list(k)

                    mb_bonds.append(edge)
                    mb_diheds.append(dihed)

        # Build bonds/dihedrals
        n_mb_bonds = len(mb_bonds)
        n_mb_diheds = len(mb_diheds)

        n_tot_bonds = n_mb_bonds + self.n_chrom * (self.n_beads - 1)
        n_pl_angles = self.n_chrom * (self.n_beads - 2)

        snap.bonds.resize(n_tot_bonds)
        snap.angles.resize(n_pl_angles)
        snap.dihedrals.resize(n_mb_diheds)

        for i in range(n_mb_bonds):
            snap.bonds.group[i] = mb_bonds[i]
            snap.bonds.typeid[i] = 0

        for i in range(n_mb_diheds):
            snap.dihedrals.group[i] = mb_diheds[i]
            snap.dihedrals.typeid[i] = 0

        for i in range(self.n_chrom):
            for j in range(self.n_beads - 1):
                k = n_mb_bonds + i * (self.n_beads - 1) + j
                l = k - n_mb_bonds + self.n_vert + i

                snap.bonds.group[k] = [l, l + 1]
                snap.bonds.typeid[k] = 1

        for i in range(self.n_chrom):
            for j in range(self.n_beads - 2):
                k = i * (self.n_beads - 2) + j
                l = k + self.n_vert + 2 * i

                snap.angles.group[k] = [l, l + 1, l + 2]
                snap.angles.typeid[k] = 0

        # Save simplices to file for visualisation
        np.savetxt(self.path_simp, hull.simplices)


    def kg_func(self, theta, kappa):
        """Kremer-Grest bending energy penalty function"""

        V = kappa * (1. + np.cos(theta))
        T = kappa * np.sin(theta)

        return V, T


    def poly_func(self, r, rmin, rmax, epsilon):
        """Polynomial (soft) pair excluded volume potential"""

        term1 = (r* np.sqrt(6/7)) / rmax
        term2 = 1 + (term1**12) * (term1**2 - 1) * (823543/46656)
		
        V = epsilon * term2
        F = -epsilon * (12.0*r**13/rmax**14 + 84.0*r**11*(0.857142857142857*r**2/rmax**2 - 1)/rmax**12)
		
        return V, F


    def set_relax_field(self, nc=10, A=45.):
        """Setup uniform membrane vertex distribution through Thompson relaxation"""

        rc = nc * self.r0
        Ac = A * self.r0

        md.constrain.sphere(group=hoomd.group.charged(), P=(0, 0, 0), r=self.r_memb)

        # Set electrostatic repulsion between membrane beads only
        nl = md.nlist.cell()
        el = md.pair.dipole(r_cut=rc, nlist=nl)

        el.pair_coeff.set(['membrane'], ['membrane'], mu=0., A=Ac, kappa=0.)
        el.pair_coeff.set(['poly_soft', 'poly_stiff', 'poly_growth'], ['membrane', 'poly_soft', 'poly_stiff', 'poly_growth'],
                          mu=0., A=0., kappa=0., r_cut=0)


    def set_full_field(self):
        """Full force field parameters"""

        rc = max(self.r0, self.rm_wca)
        nl = md.nlist.cell() if self.nlist == "cell" else md.nlist.tree()

        # Set WCA repulsion
        self.lj_pp = md.pair.table(width=1000, nlist=nl)
        self.lj_pm = md.pair.lj(r_cut=2 ** (1 / 6.) * rc, nlist=nl, name="pm")

        self.lj_pm.set_params(mode='shift')

        self.lj_pp.pair_coeff.set(['poly_soft', 'poly_stiff', 'poly_growth'], ['poly_soft', 'poly_stiff', 'poly_growth'], func=self.poly_func, rmin=0., rmax=1.0, coeff=dict(epsilon=5))
        self.lj_pp.pair_coeff.set(['membrane'], ['membrane'], func=self.poly_func, rmin=0, rmax=1.0, coeff=dict(epsilon=0))
        self.lj_pp.pair_coeff.set(['poly_soft', 'poly_stiff', 'poly_growth'], ['membrane'], func=self.poly_func, rmin=0, rmax=1.0, coeff=dict(epsilon=0))

        self.lj_pm.pair_coeff.set(['poly_soft', 'poly_stiff', 'poly_growth'], ['poly_soft', 'poly_stiff', 'poly_growth'], epsilon=0., sigma=1., r_cut=0)
        self.lj_pm.pair_coeff.set(['membrane'], ['membrane'], epsilon=0., sigma=1., r_cut=0)
        self.lj_pm.pair_coeff.set(['poly_soft', 'poly_stiff', 'poly_growth'], ['membrane'], epsilon=1., sigma=rc)

        # Set non-specific attraction
        self.lj_at = md.pair.lj(r_cut=self.rc_at, nlist=nl, name="pp")

        self.lj_at.pair_coeff.set(['poly_soft', 'poly_stiff', 'poly_growth'], ['poly_soft', 'poly_stiff', 'poly_growth'], epsilon=self.D0, sigma=1.)
        self.lj_at.pair_coeff.set(['membrane'], ['poly_soft', 'poly_stiff', 'poly_growth', 'membrane'], epsilon=0., sigma=1.)

        # Set bonded potentials
        self.bond_harm = md.bond.harmonic()

        self.angle_kg = md.angle.table(width=1000)
        self.dihed_harm = md.dihedral.harmonic()

        self.bond_harm.bond_coeff.set('mb_bonds', k=self.k, r0=self.r0)
        self.bond_harm.bond_coeff.set('pl_bonds', k=self.k_stretch, r0=1.)

        self.angle_kg.angle_coeff.set('pl_soft', func=self.kg_func, coeff=dict(kappa=self.lp_soft))
        self.angle_kg.angle_coeff.set('pl_stiff', func=self.kg_func, coeff=dict(kappa=self.lp_stiff))
        self.angle_kg.angle_coeff.set('pl_growth', func=self.kg_func, coeff=dict(kappa=self.lp_soft))

        self.dihed_harm.dihedral_coeff.set('mb_diheds', k=self.k_dihed, d=1, n=1)


    def init_stiff(self, r=5):
        snap = self.system.take_snapshot(all=True)
        com = snap.particles.position.mean(axis=0)

        for i in range(self.n_chrom):
            for j in range(self.n_beads - 2):
                k = i * (self.n_beads - 2) + j
                l = k + self.n_vert + 2 * i

                if np.linalg.norm(snap.particles.position[l + 1] - com) < r:
                    snap.angles.typeid[k] = 2
                    snap.particles.typeid[l + 1] = 3

        self.system.restore_snapshot(snap)


    def grow_stiff(self, delta=15):
        snap = self.system.take_snapshot(all=True)
        ids = np.nonzero(snap.particles.typeid > 1)[0]

        domains = np.split(ids, np.where(np.diff(ids) != 1)[0]+1)

        for d in domains:
            for l in d:
                k = l - 1 - self.n_vert

                snap.angles.typeid[k] = 1
                snap.particles.typeid[l] = 2

        for d in domains:
            for l in range(d[0] - delta, d[-1] + delta + 1):
                if (l > self.n_vert) & (l < self.n_tot - 1):
                    k = l - 1 - self.n_vert

                    if snap.angles.typeid[k] == 0:
                        snap.angles.typeid[k] = 2
                        snap.particles.typeid[l] = 3

        self.system.restore_snapshot(snap)


    def set_output(self, period=1e6):
        """Setup trajectories, logs and restart files"""

        hoomd.dump.gsd(self.path_traj,
                       group=hoomd.group.all(),
                       period=1e4 if not self.relaxed else period, phase=0,
                       overwrite=self.overwrite)

        if not self.relaxed:
            hoomd.analyze.log(filename=self.path_ana,
                              quantities=['kinetic_energy',
                                          'potential_energy'],
                              period=1e3, phase=0,
                              overwrite=True)

        else:
            hoomd.dump.gsd(filename=self.path_rest,
                           group=hoomd.group.all(),
                           truncate=True,
                           period=period, phase=0)

            hoomd.analyze.log(filename=self.path_ana,
                              quantities=['kinetic_energy',
                                          'potential_energy',
                                          'pair_lj_energy_pm',
                                          'pair_table_energy',
                                          'bond_harmonic_energy',
                                          'angle_table_energy',
                                          'dihedral_harmonic_energy'],
                              period=1e5, phase=0,
                              overwrite=self.overwrite)


    def log(self, str, color):
        """Colored file logger"""

        # ANSI colors
        CSI = "\033["
        reset = CSI + "0m\n"

        self.file_log.write(CSI + ANSIColor[color] + str + reset)
        self.file_log.flush()


    def run(self, steps, dt=5e-4,
			lpmin=1, lpmax=50, kmin=6.49, kmax=2500, dmin=1e-2, dmax=0.25, fmin=1e-2, fmax=2e5, rmin=1, rmax=4,
			num=60+1):
        """Setup integrators and run"""

        if not self.relaxed:
            # Relax membrane beads through the Fast Inertial Relaxation Engine
            md.integrate.mode_minimize_fire(dt=1e-4 * self.r0,
                                            ftol=1e-1 / self.r0 ** 2, Etol=1e-5 / self.r0, aniso=False)
            md.integrate.nve(group=hoomd.group.charged(), limit=self.r0 / 100.)

            hoomd.run(1e6)

        else:
            dt = 5e-6 if self.decondense else dt
        
            rng_seed = os.urandom(4)
            rng_seed = int(codecs.encode(rng_seed, 'hex'), 16)

            md.integrate.mode_standard(dt=dt)
            md.integrate.langevin(group=hoomd.group.all(), kT=1.0, seed=rng_seed)

            # Parameter sweeps
            if self.sweep:
                ls = np.logspace(np.log10(lpmin), np.log10(lpmax), num=num)
                ks = np.logspace(np.log10(kmax), np.log10(kmin), num=num)
                ds = np.logspace(np.log10(dmin), np.log10(self.D0), num=num)
                fs = np.logspace(np.log10(fmin), np.log10(fmax), num=num)
                rs = np.linspace(rmin, rmax, num=num)

                for idx in range(num):
                    self.log("Starting run %d out of %d" % (idx + 1, num), 'cya')

                    if "lp_soft" in self.sweep:
                        self.lp_soft = ls[idx]

                        self.angle_kg.angle_coeff.set('pl_soft', func=self.kg_func, coeff=dict(kappa=self.lp_soft))

                        self.log("Chromatin soft persistence length: %.3f" % self.lp_soft, 'blu')

                    if "lp_stiff" in self.sweep:
                        self.lp_stiff = ls[idx]

                        self.angle_kg.angle_coeff.set('pl_stiff', func=self.kg_func, coeff=dict(kappa=self.lp_stiff))

                        self.log("Chromatin stiff persistence length: %.3f" % self.lp_stiff, 'blu')

                    if "r" in self.sweep:
                        self.r0 = rs[idx]
                        rc = max(self.r0, self.rm_wca)

                        self.bond_harm.bond_coeff.set('mb_bonds', r0=self.r0)
                        self.lj_pm.pair_coeff.set(['poly_soft', 'poly_stiff', 'poly_growth'], ['membrane'],
                                                  sigma=rc, r_cut=2 ** (1 / 6.) * rc)

                        self.log("Membrane equilibrium bond length: %.3f" % self.r0, 'blu')

                    if "k" in self.sweep:
                        self.k = ks[idx]

                        self.bond_harm.bond_coeff.set('mb_bonds', k=self.k)

                        self.log("Membrane Young modulus: %.3f" % self.k, 'blu')

                    if "D" in self.sweep:
                        self.D0 = ds[idx]

                        self.lj_at.pair_coeff.set(['poly_soft', 'poly_stiff', 'poly_growth'], ['poly_soft', 'poly_stiff', 'poly_growth'],
                                                  epsilon=self.D0)

                        self.log("Non-specific potential depth: %.3f" % self.D0, 'blu')

                    if "force" in self.sweep:
                        r_exc = rs[idx]
                        rc = max(r_exc, self.rm_wca) # Prevent polymer from crossing the membrane for large shell deformations
                        
                        self.lj_pm.pair_coeff.set(['poly_soft', 'poly_stiff', 'poly_growth'], ['membrane'],
                                                  sigma=rc, r_cut=2 ** (1 / 6.) * rc)
                                                                          
                        f = fs[idx]

                        pull_npole = md.force.constant(fx=+self.dir[0]*f, fy=+self.dir[1]*f, fz=+self.dir[2]*f, group=self.npole)
                        pull_spole = md.force.constant(fx=-self.dir[0]*f, fy=-self.dir[1]*f, fz=-self.dir[2]*f, group=self.spole)
                        
                        pull_ngroup = md.force.constant(fx=+self.dir[0]*f/2, fy=+self.dir[1]*f/2, fz=+self.dir[2]*f/2, group=self.ngroup)
                        pull_sgroup = md.force.constant(fx=-self.dir[0]*f/2, fy=-self.dir[1]*f/2, fz=-self.dir[2]*f/2, group=self.sgroup)
						
                        self.log("Stretching force: %.3f" % f, 'blu')

                    hoomd.run(steps)
                    
                    if "force" in self.sweep:
                        pull_npole.disable()
                        pull_spole.disable()
                        
                        pull_ngroup.disable()
                        pull_sgroup.disable()

            elif self.propagate:
                n_growth = 50

                self.init_stiff()

                ls = np.linspace(lpmin, lpmax, num=num)

                self.log("Starting gradual stiffening run...", 'blu')

                self.angle_kg.angle_coeff.set('pl_stiff', func=self.kg_func, coeff=dict(kappa=lpmax))

                for idx in range(num):
                    lp_acet = ls[idx]

                    self.angle_kg.angle_coeff.set('pl_growth', func=self.kg_func, coeff=dict(kappa=lp_acet))
                    self.log("Acetylating chromatin persistence length: %.3f" % lp_acet, 'whi')

                    hoomd.run(steps)

                for i in range(n_growth):
                    self.log("Starting growth run %d out of %d..." % (i+1, n_growth), 'blu')

                    self.grow_stiff()

                    for idx in range(num):
                        lp_acet = ls[idx]

                        self.angle_kg.angle_coeff.set('pl_growth', func=self.kg_func, coeff=dict(kappa=lp_acet))
                        self.log("Acetylating chromatin persistence length: %.3f" % lp_acet, 'whi')

                        hoomd.run(steps)

            # Single run
            else:
                if self.decondense:
                    rc = max(rmax, self.rm_wca) # Prevent polymer from crossing the membrane for large shell deformations
                    
                    self.bond_harm.bond_coeff.set('mb_bonds', k=kmin)
                    self.lj_pm.pair_coeff.set(['poly_soft', 'poly_stiff', 'poly_growth'], ['membrane'],
									  sigma=rc, r_cut=2 ** (1 / 6.) * rc)
									  
                hoomd.run(steps)

        # Dump final frame to file
        hoomd.dump.gsd(filename=self.path_rest if self.relaxed else self.path_rel,
                       group=hoomd.group.all(),
                       period=None, overwrite=True)


# ********************** #

if __name__ == "__main__":
    sim = HoomdSim(**vars(parser.parse_args()))
    
    sim.build()
    
    if sim.propagate:
        sim.run(1e6)
    elif sim.decondense:
        sim.run(1e9)
    else:
        sim.run(1e8)
