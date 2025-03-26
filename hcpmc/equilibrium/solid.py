# solid.py
# Author: Jin Wang
"""Provide two calculation of statistical property of solid composed of hard polyhedron particles.

Now supportion: Pressure, Free Energy (Several Type Lattice).

.. invisible-code-block: python

"""

import hoomd
import numpy as np
import pandas as pd
import os
import datetime
import matplotlib.pyplot as plt
import gsd


class Pressure:
    """Calculate the pressure of a solid sample composed of hard particles with given particlefactory."""

    def __init__(self, samplename, factory):
        self.samplename = samplename
        self.particlefactory = factory
        self.pressure = []
        self.packingfraction = None

    def calculate(self, n_equili: int = 100000, n_sampling: int = 10000, seed: int = 12345):
        """Calculate the pressure of a solid composed of hard polyhedron particles.

        System will equilibrate for n_equili steps and calculate pressure for n_sampling times.
        """
        with gsd.hoomd.open(self.samplename) as f:
            N = f[0].particles.N
            box = f[0].configuration.box
            V = box[0] * box[1] * box[2]
            rho = N/V
        print(f"Start calculating... packing fraction: {rho:.4f}", flush=True)
        start_time = datetime.datetime.now()
        sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=seed)
        sim.create_state_from_gsd(self.samplename)
        mc = self.particlefactory.get_integrator()
        sim.operations.integrator = mc
        sim.run(0)
        if mc.overlaps != 0:
            print("Overlaps found in the initial state")
            return 0

        tune = hoomd.hpmc.tune.MoveSize.scale_solver(
            moves=["a", "d"],
            target=0.3,
            trigger=hoomd.trigger.Before(n_equili // 2),
            max_translation_move=1,
            max_rotation_move=1,
        )
        sim.operations.tuners.append(tune)
        Boxmc = hoomd.hpmc.update.BoxMC(trigger=hoomd.trigger.Periodic(1), P=-1)
        Boxmc.aspect.update({"delta": 0.001, "weight": 1.0})
        Boxmc.shear.update({"weight": 1.0, "delta": (0.001, 0.001, 0.001)})
        sim.operations.updaters.append(Boxmc)
        sdf = hoomd.hpmc.compute.SDF(0.02, 1e-4)
        sim.operations.computes.append(sdf)
        logger = hoomd.logging.Logger()
        logger.add(mc, quantities=["type_shapes"])
        logger.add(sdf, quantities=["betaP"])
        gsd_writer = hoomd.write.GSD(
            filename="equilibrium.gsd",
            trigger=hoomd.trigger.Periodic(int(5e3)),
            logger=logger,
            mode="xb",
        )
        sim.operations.writers.append(gsd_writer)

        time0 = sim.timestep
        while sim.timestep - time0 < n_equili:
            sim.run(5000)
            duration = datetime.datetime.now() - start_time
            formatted_duration = str(duration).split(".")[0]

            print(f"{formatted_duration}  Equilibrating...({sim.timestep - time0}/{n_equili + n_sampling}) {sdf.betaP}", flush=True)

        logger2 = hoomd.logging.Logger()
        logger2.add(sdf, quantities=["betaP"])
        gsd_writer2 = hoomd.write.GSD(
            filename="pressure.gsd",
            trigger=hoomd.trigger.Periodic(1),
            logger=logger2,
            dynamic=[],
            filter=hoomd.filter.Null(),
            mode="xb",
        )
        sim.operations.writers.append(gsd_writer2)
        while sim.timestep - time0 < n_sampling + n_equili:
            sim.run(5000)
            duration = datetime.datetime.now() - start_time
            formatted_duration = str(duration).split(".")[0]

            print(f"{formatted_duration} Sampling... ({sim.timestep - time0}/{n_equili + n_sampling}) {sdf.betaP}", flush=True)

        gsd_writer.flush()
        gsd_writer2.flush()
        self.packingfraction = sim.state.N_particles / sim.state.box.volume
        df = pd.DataFrame(gsd.hoomd.read_log("pressure.gsd", scalar_only=True))
        betaP = df["log/hpmc/compute/SDF/betaP"]
        self.pressure = betaP.iloc[-n_sampling:]
        print(f"pressure = {np.mean(self.pressure)} +/- {np.std(self.pressure) / np.sqrt(len(self.pressure))}")

    def get_pressure(self):
        """Get the pressure of a solid composed of hard polyhedron particles."""
        return f"{np.mean(self.pressure)} +/- {np.std(self.pressure) / np.sqrt(len(self.pressure))}"


class Harmonic:
    """Calculate the free energy derivative in Harmonic Potential.

    Use Frenkel-Ladd method to calculate the free energy.
    """

    def __init__(self, samplename, factory, k):
        self.samplename = samplename
        self.particlefactory = factory
        self.dF = None
        self.k = k

    def calculate(self, n_equili: int = 20000, n_sampling: int = 30000, seed: int = 12345):
        """Calculate the free energy derivative.

        System will equilibrate for n_equili steps and calculate dF for n_sampling times with given k.
        """
        start_time = datetime.datetime.now()
        sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=seed)
        state_fn = self.samplename
        with gsd.hoomd.open(name=state_fn) as f:
            ref_positions = f[0].particles.position
            ref_orientations = f[0].particles.orientation
        sim.create_state_from_gsd(state_fn)
        mc = self.particlefactory.get_integrator()
        sim.operations.integrator = mc
        sim.run(0)
        if mc.overlaps != 0:
            print("Overlaps found in the initial state")
            return 0

        f = gsd.hoomd.open(self.samplename)
        N = f[0].particles.N

        tune = hoomd.hpmc.tune.MoveSize.scale_solver(
            moves=["a", "d"],
            target=0.3,
            trigger=2000,
            max_translation_move=1,
            max_rotation_move=1,
        )
        sim.operations.tuners.append(tune)
        RD = hoomd.update.RemoveDrift(ref_positions, trigger=1)
        sim.operations.updaters.append(RD)
        symmetries = self.particlefactory.get_symmetries()
        Har = hoomd.hpmc.external.Harmonic(
            reference_positions=ref_positions,
            reference_orientations=ref_orientations,
            k_translational=self.k,
            k_rotational=self.k,
            symmetries=symmetries,
        )
        sim.operations.integrator.external_potentials.append(Har)

        logger2 = hoomd.logging.Logger()
        logger2.add(Har, quantities=["energy"])
        gsd_writer = hoomd.write.GSD(
            filename="Harmonic.gsd",
            trigger=1,
            mode="wb",
            dynamic=[],
            filter=hoomd.filter.Null(),
            logger=logger2,
        )
        sim.operations.writers.append(gsd_writer)
        time0 = sim.timestep
        while sim.timestep - time0 < n_equili + n_sampling:
            sim.run(5000)
            duration = datetime.datetime.now() - start_time
            formatted_duration = str(duration).split(".")[0]

            print(f"{formatted_duration}  Calculating dF for k = {self.k}... ({sim.timestep - time0}/{n_equili + n_sampling})", flush=True)

        gsd_writer.flush()
        df = pd.DataFrame(gsd.hoomd.read_log("Harmonic.gsd", scalar_only=True))
        energy = df["log/hpmc/external/Harmonic/energy"]
        self.dF = energy.iloc[-n_sampling:].mean() / N
        print(f"dF = {self.dF:.3f}")

        return self.dF


class FreeEnergy:
    """Calculate the free energy of a solid sample composed of hard particles with given particlefactory.

    Use Frenkel-Ladd method to calculate the free energy.

    Denser sample will need larger kmax and n_k.
    """

    def __init__(self, samplename, factory):
        self.samplename = samplename
        self.particlefactory = factory
        self.dF = []
        self.ks = np.exp(np.linspace(np.log(2e4), np.log(2e-4), 20))
        self.F = None
        self.Fex = None

    def set_ks(self, kmin: float, kmax: float, n_k: int):
        """Set ks."""
        self.ks = np.exp(np.linspace(np.log(kmax), np.log(kmin), n_k))

    def calculate(self, seed: int = 12345, n_equili: int = 20000, n_sampling: int = 30000):
        """Calculate the pressure of a solid composed of hard polyhedron particles.

        System will equilibrate for n_equili steps and calculate dF for n_sampling times with every lamda.

        Totally n_sampling * n_k steps will be run.
        """
        start_time = datetime.datetime.now()
        sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=seed)
        state_fn = self.samplename
        with gsd.hoomd.open(name=state_fn) as f:
            ref_positions = f[0].particles.position
            ref_orientations = f[0].particles.orientation
        sim.create_state_from_gsd(state_fn)
        mc = self.particlefactory.get_integrator()
        sim.operations.integrator = mc
        sim.run(0)
        if mc.overlaps != 0:
            print("Overlaps found in the initial state")
            return 0

        f = gsd.hoomd.open(self.samplename)
        N = f[0].particles.N
        box = f[0].configuration.box
        V = box[0] * box[1] * box[2]
        Nsym = len(self.particlefactory.get_symmetries())

        tune = hoomd.hpmc.tune.MoveSize.scale_solver(
            moves=["a", "d"],
            target=0.3,
            trigger=2000,
            max_translation_move=1,
            max_rotation_move=1,
        )
        sim.operations.tuners.append(tune)
        RD = hoomd.update.RemoveDrift(ref_positions, trigger=1)
        sim.operations.updaters.append(RD)
        symmetries = self.particlefactory.get_symmetries()
        Har = hoomd.hpmc.external.Harmonic(
            reference_positions=ref_positions,
            reference_orientations=ref_orientations,
            k_translational=-1,
            k_rotational=-1,
            symmetries=symmetries,
        )
        sim.operations.integrator.external_potentials.append(Har)

        logger2 = hoomd.logging.Logger()
        logger2.add(Har, quantities=["energy"])
        for i, k in enumerate(self.ks):
            duration = datetime.datetime.now() - start_time
            formatted_duration = str(duration).split(".")[0]
            print(f"{formatted_duration}  Calculating dF for k = {k}... ({i + 1}/{len(self.ks)})", flush=True, end="  ")
            tempfile = f"Fenergy_temp_{i}.gsd"
            gsd_writer = hoomd.write.GSD(
                filename=tempfile,
                trigger=1,
                mode="wb",
                dynamic=[],
                filter=hoomd.filter.Null(),
                logger=logger2,
            )
            sim.operations.writers.append(gsd_writer)

            Har.k_translational = k
            Har.k_rotational = k
            sim.run(n_equili)
            sim.run(n_sampling)
            gsd_writer.flush()
            df = pd.DataFrame(gsd.hoomd.read_log(tempfile, scalar_only=True))
            energy = df["log/hpmc/external/Harmonic/energy"]
            self.dF.append(energy.iloc[-n_sampling:].mean() / N)

            print(f"dF = {self.dF[-1]:.3f}")
            sim.operations.writers.pop()
            os.remove(tempfile)

        integral = -np.trapezoid(self.dF, np.log(self.ks))
        F_Ein = -3 / 2 * (N - 1) * np.log(np.pi / (self.ks[0] / 2)) - 3 / 2 * N * np.log(np.pi / (self.ks[0] / 2)) + N * np.log(2 * np.pi**2) - N * np.log(Nsym)
        dF_CM = -np.log(N / V) + 3 / 2 * np.log(N)
        F_ig = N * np.log(N / V) - N
        self.F = (F_Ein - integral * N - dF_CM) / N
        self.Fex = (self.F - F_ig) / N

    def plot_df(self):
        """Plot the dF to check if the values of k is appropriate."""
        plt.plot(self.ks, self.dF, "o")
        plt.xscale("log")
        plt.show()

    def get_free_energy(self):
        """Get the free energy per particle of a solid composed of hard polyhedron particles."""
        return self.F

    def get_excess_free_energy(self):
        """Get the free energy per particle of a solid composed of hard polyhedron particles."""
        return self.Fex
