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
from .. import initializer


class Pressure:
    """Calculate the pressure of a fluid sample composed of hard particles with given particlefactory."""

    def __init__(self, samplename, sample0name, factory):
        self.samplename = samplename
        self.sample0name = sample0name
        self.particlefactory = factory
        self.pressure = []
        self.packingfraction = None

    def calculate(self, betaP_set: float, n_equili: int = 100000, n_sampling: int = 10000, seed: int = 12345):
        """Calculate the pressure of a solid composed of hard polyhedron particles.

        System will equilibrate for n_equili steps and calculate pressure for n_sampling times.
        """
        print(f"Start calculating... betaP target: {betaP_set}")
        start_time = datetime.datetime.now()
        sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=seed)
        sim.create_state_from_gsd(self.sample0name)
        mc = self.particlefactory.get_integrator()
        sim.operations.integrator = mc
        sim.run(0)
        if mc.overlaps != 0:
            print("Overlaps found in the initial state")
            return 0
        tune = hoomd.hpmc.tune.MoveSize.scale_solver(
            moves=["a", "d"],
            target=0.5,
            trigger=200,
            max_translation_move=0.3,
            max_rotation_move=1,
        )
        sim.run(1e4)
        sim.operations.tuners.append(tune)
        sdf = hoomd.hpmc.compute.SDF(0.02, 1e-4)
        sim.operations.computes.append(sdf)
        logger = hoomd.logging.Logger()
        logger.add(mc, quantities=["type_shapes"])
        logger.add(sdf, quantities=["betaP"])
        gsd_writer = hoomd.write.GSD(
            filename=self.samplename,
            trigger=hoomd.trigger.Periodic(int(5e3)),
            logger=logger,
            mode="xb",
        )
        sim.operations.writers.append(gsd_writer)

        time0 = sim.timestep
        Boxmc = hoomd.hpmc.update.BoxMC(1, betaP_set)
        Boxmc.volume.update({"weight": 1.0, "delta": 0.02 * sim.state.box.volume})
        sim.operations.updaters.append(Boxmc)

        while sim.timestep - time0 < n_equili:
            sim.run(10)
            Boxmc.volume.update({"weight": 1.0, "delta": 0.02 * sim.state.box.volume})
            duration = datetime.datetime.now() - start_time
            formatted_duration = str(duration).split(".")[0]

            print(f"{formatted_duration} Equilibrating... ({sim.timestep - time0}/{n_equili + n_sampling}) {sdf.betaP}", flush=True)

        sim.operations.updaters.remove(Boxmc)

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
            success_ratio = fv.free_volume / sim.state.box.volume
            self.chemicalpotential.append(np.log(rho) - np.log(success_ratio))

    def get_chemical_potential(self):
        """Get the chemical_potential of a fluid composed of hard polyhedron particles."""
        return f"{np.mean(self.chemicalpotential)} +/- {np.std(self.chemicalpotential) / np.sqrt(len(self.chemicalpotential))}"
