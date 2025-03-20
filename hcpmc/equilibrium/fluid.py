# fluid.py
# Author: Jin Wang
"""Provide two calculation of statistical property of fluid composed of hard polyhedron particles.

Now supportion: Pressure, ChemicalPotential.

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
    """Calculate the pressure of a fluid sample composed of hard particles with given particlefactory."""

    def __init__(self, samplename, factory):
        self.samplename = samplename
        self.particlefactory = factory
        self.pressure = []

    def calculate(self, n_equili: int = 100000, n_sampling: int = 10000, seed: int = 12345):
        """Calculate the pressure of a solid composed of hard polyhedron particles.

        System will equilibrate for n_equili steps and calculate pressure for n_sampling times.
        """
        print("Start calculating...")
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
        sdf = hoomd.hpmc.compute.SDF(0.02, 1e-4)
        sim.operations.computes.append(sdf)
        logger = hoomd.logging.Logger()
        logger.add(mc, quantities=["type_shapes"])
        logger.add(sdf,quantities=['betaP'])
        gsd_writer = hoomd.write.GSD(
            filename="equilibrium.gsd",
            trigger=hoomd.trigger.Periodic(int(5e3)),
            mode="xb",
        )
        sim.operations.writers.append(gsd_writer)
        time0 = sim.timestep
        while sim.timestep < n_equili:
            sim.run(5000)
            duration = datetime.datetime.now() - start_time
            formatted_duration = str(duration).split(".")[0]

            print(f"{formatted_duration} Equilibrating... ({sim.timestep - time0}/{n_equili + n_sampling*10})", flush=True)

        for i in range(n_sampling):
            sim.run(10)
            self.pressure.append(sdf.betaP)
            duration = datetime.datetime.now() - start_time
            formatted_duration = str(duration).split(".")[0]
            print(f"{formatted_duration}  Sampling...({sim.timestep - time0}/{n_equili + n_sampling*10}) {sdf.betaP}", flush=True)

    def get_pressure(self):
        """Get the pressure of a fluid composed of hard polyhedron particles."""
        return f"{np.mean(self.pressure)} +/- {np.std(self.pressure) / np.sqrt(len(self.pressure))}"


class ChemicalPotential:
    """Calculate the chemical potential of a fluid sample composed of hard polyhedron particles.

    Use Widom-insertion method.
    """

    def __init__(self, samplename, factory):
        self.samplename = samplename
        self.particlefactory = factory
        self.chemicalpotential = []

    def calculate(self, n_sampling: int = 10000, n_trials: int = 100000, seed: int = 12345):
        """Calculate the chemical potential of a solid composed of hard polyhedron particles.

        System will equilibrate for 1000 steps and calculate ChemicalPotential for n_sampling times.

        System will try inserting one particle then remove for n_trials times.

        Totally n_sampling * n_trials calculation.
        """
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
            trigger=hoomd.trigger.Before(10000),
            max_translation_move=1,
            max_rotation_move=1,
        )
        sim.run(1e4)
        fv = hoomd.hpmc.compute.FreeVolume(test_particle_type="S0", num_samples=n_trials)
        sim.operations.computes.append(fv)
        rho = sim.state.N_particles / sim.state.box.volume
        for i in range(n_sampling):
            sim.run(10)
            success_ratio=fv.free_volume / sim.state.box.volume
            self.chemicalpotential.append(np.log(rho)-np.log(success_ratio))

    def get_chemical_potential(self):
        """Get the chemical_potential of a fluid composed of hard polyhedron particles."""
        return f"{np.mean(self.chemicalpotential)} +/- {np.std(self.chemicalpotential) / np.sqrt(len(self.chemicalpotential))}"
