# Initializer.py
# Author: Jin Wang
"""A 'Initializer' class can generate initial state(frame) in 'gsd' format.

Now we support the following options:
Densest, SC, FCC, BCC, HCP

.. invisible-code-block: python

    factory = hcpmc.particlefactory.Family423(a=1,c=2)
    dense = hcpmc.initializer.Densest(factory = factory, particle_per_cell=1, filename='sample.gsd')
    dense.generate(RecallTimes = 50)
    densestPF = dense.get_dpf()
    dense.create_sample(0.7, [2,2,2])

"""

import hoomd
from . import particlefactory
import numpy as np
import datetime
import gsd.hoomd
import os
import shutil
import signac
from flow import FlowProject
import subprocess


class Expander:
    """An class that could expand sample0."""

    def __init__(self, factory: particlefactory, sample0name: str):
        self.particlefactory = factory
        self.sample0 = sample0name

    def get_sample0(self):
        """Get the sample."""
        filename = self.sample0
        return filename

    def create_sample(self, packingfraction, replication, samplename):
        """create_sample can create sample with given packingfraction and replication."""
        f = self.sample0
        sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=np.random.randint(65536))
        sim.create_state_from_gsd(f)
        mc = self.particlefactory.get_integrator()
        sim.operations.integrator = mc
        sim.run(0)
        logger = hoomd.logging.Logger()
        try:
            logger.add(mc, quantities=["type_shapes"])
        except:
            print("no type_shapes")
        pf = sim.state.N_particles / (sim.state.box.volume)
        scale = (pf / packingfraction) ** (1 / 3)
        box0 = sim.state.box
        box = [
            box0.Lx * scale,
            box0.Ly * scale,
            box0.Lz * scale,
            box0.xy,
            box0.xz,
            box0.yz,
        ]
        hoomd.update.BoxResize.update(state=sim.state, box=box)
        nx = int(replication[0])
        ny = int(replication[1])
        nz = int(replication[2])
        sim.state.replicate(nx=nx, ny=ny, nz=nz)
        sim.run(0)
        hoomd.write.GSD.write(
            state=sim.state,
            mode="wb",
            filename=samplename,
            filter=hoomd.filter.All(),
            logger=logger,
        )



class Densest(Expander):
    """class for create the densest packed (namely Close-Packed) sample.

    Put in particle factory, the number of particles per cell and sample filename.

    MUST call generate several times before create_sample.
    """

    def __init__(self, factory: particlefactory, particle_per_cell: int, sample0name: str, seeds: list):
        self.particlefactory = factory
        self.particle_per_cell = particle_per_cell
        self.seeds = seeds
        self.sample0 = sample0name

        self.densestpath = None
        self.densest = None
        # initial signac project
        self.folder = "./DensestProject"
        try:
            os.mkdir(self.folder)
        except FileExistsError:
            pass
        self.project = FlowProject.init_project(self.folder)
        src = os.path.dirname(__file__) + "/utils/Densest.py"
        dst = self.folder + "/project.py"
        if os.path.isfile(src):
            shutil.copy(src, dst)
        for seed in seeds:
            sp = dict(mc_vertices=factory.shape.vertices, particle_per_cell=particle_per_cell, seed=seed)
            self.project.open_job(sp).init()

    def compress(self):
        """Try to generate the densest packing crystal gsd sample.

        Warning: One calling is not neccessary densest. Recommend to call this several times (better parallelly).
        """
        frame = gsd.hoomd.Frame()
        frame.particles.N = self.particle_per_cell
        position = []
        for j in range(self.particle_per_cell):
            pos = [0, 0, 10 * j - self.particle_per_cell // 2 * 10]
            position.append(pos)
        frame.particles.position = position
        frame.particles.orientation = [1, 0, 0, 0] * self.particle_per_cell
        frame.particles.typeid = [0] * self.particle_per_cell
        frame.particles.types = ["S0"]
        frame.configuration.box = [10, 10, self.particle_per_cell * 10, 0, 0, 0]
        with gsd.hoomd.open(self.folder + "/dense_temp.gsd", "w") as f:
            f.append(frame)

        result = subprocess.run(["python", "project.py", "submit", "-o", "compress"], capture_output=True, text=True, cwd=self.folder)
        print("Output:", result.stdout)
        print("Error:", result.stderr)
        print("Exit Code:", result.returncode)
        print("Check Status Command:")
        print("cd ", os.getcwd() + self.folder[1:])
        print("watch -n 1 python3 project.py status")

    def analysis(self):
        """Get the densest packing fraction.

        Must After finish compress.
        """
        MAX = -1
        for job in self.project:
            try:
                if job.doc["phi"] > MAX:
                    MAX = job.doc["phi"]
                    self.densestpath = job.path
                    self.densest = job.doc["phi"]
            except KeyError:
                continue
        return MAX

    def create_sample0(self):
        """Get the densest packing fraction sample0.

        Must After call compress and analysis.
        """
        if self.densest == [-1]:
            raise ValueError("The densest packing fraction has not been determined yet.")

        src = self.densestpath + "/trial.gsd"
        dst = self.sample0
        try:
            shutil.copyfile(src, dst)
        finally:
            print("get successfully. packing fraction :", self.densest, "")
        return dst


class SC(Expander):
    """class for create Simple Cubic crystal sample.

    Put in particle factory and sample filename.
    """

    def __init__(self, factory: particlefactory, sample0name: str):
        self.particlefactory = factory
        self.particle_per_cell = 1
        self.packingfraction = -1
        self.sample0 = sample0name

    def create_sample0(self):
        """Get the Simple Cubic crystal sample."""
        # lattice vectors
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        v3 = np.array([0, 0, 1])
        N = self.particle_per_cell
        # box vector
        boxv1 = v1
        boxv2 = v2
        boxv3 = v3

        # compute box
        Lx = np.linalg.norm(boxv1)
        a2x = np.dot(boxv1, boxv2) / np.linalg.norm(boxv1)
        Ly = np.sqrt(np.dot(boxv2, boxv2) - a2x**2)
        xy = a2x / Ly
        Lz = np.dot(boxv3, np.cross(boxv1, boxv2) / np.linalg.norm(np.cross(boxv1, boxv2)))
        a3x = np.dot(boxv1, boxv3) / np.linalg.norm(boxv1)
        xz = a3x / Lz
        yz = (np.dot(boxv2, boxv3) - a2x * a3x) / Ly / Lz
        box = [Lx, Ly, Lz, xy, xz, yz]
        # particle positions
        particle_positions = []
        for z in range(1):
            for y in range(1):
                for x in range(1):
                    particle_positions.append(
                        [
                            x * v1[0] + y * v2[0] + z * v3[0] - boxv1[0] / 2 - boxv2[0] / 2 - boxv3[0] / 2,
                            x * v1[1] + y * v2[1] + z * v3[1] - boxv1[1] / 2 - boxv2[1] / 2 - boxv3[1] / 2,
                            x * v1[2] + y * v2[2] + z * v3[2] - boxv1[2] / 2 - boxv2[2] / 2 - boxv3[2] / 2,
                        ]
                    )
        self.packingfraction = self.particle_per_cell / (Lx * Ly * Lz)
        orientation = [(1, 0, 0, 0)] * N
        frame = gsd.hoomd.Frame()
        frame.particles.N = N
        frame.particles.position = particle_positions
        frame.particles.orientation = orientation
        frame.particles.typeid = [0] * N
        frame.particles.types = ["S0"]
        frame.particles.type_shapes = [self.particlefactory.get_gsd_shape()]
        frame.configuration.box = box
        filename = self.sample0
        with gsd.hoomd.open(name=filename, mode="w") as f:
            f.append(frame)
        return filename

    def get_pf(self):
        """Get the sample packing fraction.

        Must After call get_sample0.
        """
        if self.packingfraction == -1:
            raise ValueError("The packing fraction has not been calculated yet.")
        return self.packingfraction


class BCC(Expander):
    """class for create Body-Centered Cubic crystal sample.

    Put in particle factory and sample filename.
    """

    def __init__(self, factory: particlefactory, sample0name: str):
        self.particlefactory = factory
        self.particle_per_cell = 1
        self.packingfraction = -1
        self.sample0 = sample0name

    def create_sample0(self):
        """Get the Body-Centered Cubic crystal sample."""
        # lattice vectors
        v1 = np.array([1, 0, 0])
        v2 = np.array([0, 1, 0])
        v3 = np.array([1 / 2, 1 / 2, 1 / 2])
        N = self.particle_per_cell
        # box vector
        boxv1 = v1
        boxv2 = v2
        boxv3 = v3

        # compute box
        Lx = np.linalg.norm(boxv1)
        a2x = np.dot(boxv1, boxv2) / np.linalg.norm(boxv1)
        Ly = np.sqrt(np.dot(boxv2, boxv2) - a2x**2)
        xy = a2x / Ly
        Lz = np.dot(boxv3, np.cross(boxv1, boxv2) / np.linalg.norm(np.cross(boxv1, boxv2)))
        a3x = np.dot(boxv1, boxv3) / np.linalg.norm(boxv1)
        xz = a3x / Lz
        yz = (np.dot(boxv2, boxv3) - a2x * a3x) / Ly / Lz
        box = [Lx, Ly, Lz, xy, xz, yz]
        # particle positions
        particle_positions = []
        for z in range(1):
            for y in range(1):
                for x in range(1):
                    particle_positions.append(
                        [
                            x * v1[0] + y * v2[0] + z * v3[0] - boxv1[0] / 2 - boxv2[0] / 2 - boxv3[0] / 2,
                            x * v1[1] + y * v2[1] + z * v3[1] - boxv1[1] / 2 - boxv2[1] / 2 - boxv3[1] / 2,
                            x * v1[2] + y * v2[2] + z * v3[2] - boxv1[2] / 2 - boxv2[2] / 2 - boxv3[2] / 2,
                        ]
                    )
        self.packingfraction = self.particle_per_cell / (Lx * Ly * Lz)
        orientation = [(1, 0, 0, 0)] * N

        frame = gsd.hoomd.Frame()
        frame.particles.N = N
        frame.particles.position = particle_positions
        frame.particles.orientation = orientation
        frame.particles.typeid = [0] * N
        frame.particles.types = ["S0"]
        frame.particles.type_shapes = [self.particlefactory.get_gsd_shape()]
        frame.configuration.box = box
        filename = self.sample0
        with gsd.hoomd.open(name=filename, mode="w") as f:
            f.append(frame)
        return filename

    def get_pf(self):
        """Get the sample packing fraction.

        Must After call get_sample0.
        """
        if self.packingfraction == -1:
            raise ValueError("The packing fraction has not been calculated yet.")
        return self.packingfraction


class FCC(Expander):
    """class for create Face-Centered Cubic crystal sample.

    Put in particle factory and sample filename.
    """

    def __init__(self, factory: particlefactory, sample0name: str):
        self.particlefactory = factory
        self.particle_per_cell = 1
        self.packingfraction = -1
        self.sample0 = sample0name

    def create_sample0(self):
        """Get the Face-Centered Cubic crystal sample."""
        # lattice vectors
        v1 = np.array([1, 0, 0])
        v2 = np.array([1 / 2, np.sqrt(3) / 2, 0])
        v3 = np.array([1 / 2, np.sqrt(3) / 6, np.sqrt(6) / 3])
        N = self.particle_per_cell
        # box vector
        boxv1 = v1
        boxv2 = v2
        boxv3 = v3

        # compute box
        Lx = np.linalg.norm(boxv1)
        a2x = np.dot(boxv1, boxv2) / np.linalg.norm(boxv1)
        Ly = np.sqrt(np.dot(boxv2, boxv2) - a2x**2)
        xy = a2x / Ly
        Lz = np.dot(boxv3, np.cross(boxv1, boxv2) / np.linalg.norm(np.cross(boxv1, boxv2)))
        a3x = np.dot(boxv1, boxv3) / np.linalg.norm(boxv1)
        xz = a3x / Lz
        yz = (np.dot(boxv2, boxv3) - a2x * a3x) / Ly / Lz
        box = [Lx, Ly, Lz, xy, xz, yz]
        # particle positions
        particle_positions = []
        for z in range(1):
            for y in range(1):
                for x in range(1):
                    particle_positions.append(
                        [
                            x * v1[0] + y * v2[0] + z * v3[0] - boxv1[0] / 2 - boxv2[0] / 2 - boxv3[0] / 2,
                            x * v1[1] + y * v2[1] + z * v3[1] - boxv1[1] / 2 - boxv2[1] / 2 - boxv3[1] / 2,
                            x * v1[2] + y * v2[2] + z * v3[2] - boxv1[2] / 2 - boxv2[2] / 2 - boxv3[2] / 2,
                        ]
                    )
        self.packingfraction = self.particle_per_cell / (Lx * Ly * Lz)
        orientation = [(1, 0, 0, 0)] * N

        frame = gsd.hoomd.Frame()
        frame.particles.N = N
        frame.particles.position = particle_positions
        frame.particles.orientation = orientation
        frame.particles.typeid = [0] * N
        frame.particles.types = ["S0"]
        frame.particles.type_shapes = [self.particlefactory.get_gsd_shape()]
        frame.configuration.box = box
        filename = self.sample0
        with gsd.hoomd.open(name=filename, mode="w") as f:
            f.append(frame)
        return filename

    def get_pf(self):
        """Get the sample packing fraction.

        Must After call get_sample0.
        """
        if self.packingfraction == -1:
            raise ValueError("The packing fraction has not been calculated yet.")
        return self.packingfraction


class HCP(Expander):
    """class for create Hexagonal-Close-Packed crystal sample.

    Put in particle factory and sample filename. Note: HCP unit cell contains two particles.
    """

    def __init__(self, factory: particlefactory, sample0name: str):
        self.particlefactory = factory
        self.particle_per_cell = 2
        self.packingfraction = -1
        self.sample0 = sample0name

    def create_sample0(self):
        """Get the Hexagonal-Close-Packed crystal sample."""
        # lattice vectors
        c = np.sqrt(8 / 3)
        # lattice vectors
        v1 = np.array([1, 0, 0])
        v2 = np.array([1 / 2, np.sqrt(3), 0])
        v3 = np.array([0, 0, c])
        N = self.particle_per_cell
        # box vector
        boxv1 = v1
        boxv2 = v2
        boxv3 = v3

        # compute box
        Lx = np.linalg.norm(boxv1)
        a2x = np.dot(boxv1, boxv2) / np.linalg.norm(boxv1)
        Ly = np.sqrt(np.dot(boxv2, boxv2) - a2x**2)
        xy = a2x / Ly
        Lz = np.dot(boxv3, np.cross(boxv1, boxv2) / np.linalg.norm(np.cross(boxv1, boxv2)))
        a3x = np.dot(boxv1, boxv3) / np.linalg.norm(boxv1)
        xz = a3x / Lz
        yz = (np.dot(boxv2, boxv3) - a2x * a3x) / Ly / Lz
        box = [Lx, Ly, Lz, xy, xz, yz]
        # particle positions
        particle_positions = []
        for z in range(1):
            for y in range(1):
                for x in range(1):
                    particle_positions.append(
                        [
                            x * v1[0] + y * v2[0] + z * v3[0] - boxv1[0] / 2 - boxv2[0] / 2 - boxv3[0] / 2,
                            x * v1[1] + y * v2[1] + z * v3[1] - boxv1[1] / 2 - boxv2[1] / 2 - boxv3[1] / 2,
                            x * v1[2] + y * v2[2] + z * v3[2] - boxv1[2] / 2 - boxv2[2] / 2 - boxv3[2] / 2,
                        ]
                    )
                    particle_positions.append(
                        [
                            x * v1[0] + y * v2[0] + z * v3[0] - boxv1[0] / 2 - boxv2[0] / 2 - boxv3[0] / 2 + 1 / 2,
                            x * v1[1] + y * v2[1] + z * v3[1] - boxv1[1] / 2 - boxv2[1] / 2 - boxv3[1] / 2 + np.sqrt(3) / 6,
                            x * v1[2] + y * v2[2] + z * v3[2] - boxv1[2] / 2 - boxv2[2] / 2 - boxv3[2] / 2 + 1 / 2 * c,
                        ]
                    )
        self.packingfraction = self.particle_per_cell / (Lx * Ly * Lz)
        orientation = [(1, 0, 0, 0)] * N

        frame = gsd.hoomd.Frame()
        frame.particles.N = N
        frame.particles.position = particle_positions
        frame.particles.orientation = orientation
        frame.particles.typeid = [0] * N
        frame.particles.types = ["S0"]
        frame.particles.type_shapes = [self.particlefactory.get_gsd_shape()]
        frame.configuration.box = box
        filename = self.sample0
        with gsd.hoomd.open(name=filename, mode="w") as f:
            f.append(frame)
        return filename

    def get_pf(self):
        """Get the sample packing fraction.

        Must After call get_sample0.
        """
        if self.packingfraction == -1:
            raise ValueError("The packing fraction has not been calculated yet.")
        return self.packingfraction
