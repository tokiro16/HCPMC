# UmbrellaSampling.py
# Author: Jin Wang
"""Provide Umbrella Sampling Free Energy Calculation for solid sample. Now support only one-dimension Umbrella Sampling.

data format: Grossfield, A, “WHAM: an implementation of the weighted histogram analysis method”, http://membrane.urmc.rochester.edu/content/wham/, version XXXX

.. invisible-code-block: python
"""

from .. import OrderParameter
from . import UmbrellaWindow
from . import UmbrellaEquilibration
import os
import hoomd


class BiasSampling:
    def __init__(self, samplename, factory, k, window, SamplingNumber):
        self.samplename = samplename
        self.particlefactory = factory
        self.k = k
        self.window = window
        self.SamplingNumber = SamplingNumber

    def calculate(
        self,
        orderparameter: OrderParameter,
        simtrials: int = 50,
        equili_strict: int = 20,
        seed: int = 12345,
    ):
        sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=seed)
        state_fn = self.samplename
        mc = self.particlefactory.get_integrator()
        sim.operations.integrator = mc
        Boxmc = hoomd.hpmc.update.BoxMC(trigger=hoomd.trigger.Periodic(1), P=-1)
        Boxmc.aspect.update({"delta": 0.001, "weight": 1.0})
        Boxmc.shear.update({"weight": 1.0, "delta": (0.001, 0.001, 0.001)})
        sim.operations.updaters.append(Boxmc)
        sim.create_state_from_gsd(state_fn)
        sim.run(0)
        if mc.overlaps != 0:
            print("Overlaps found in the initial state")
            return 0
        tune = hoomd.hpmc.tune.MoveSize.scale_solver(
            moves=["a", "d"],
            target=0.3,
            trigger=2000,
            max_translation_move=1,
            max_rotation_move=1,
        )
        sim.operations.tuners.append(tune)
        OP = orderparameter
        UW = UmbrellaWindow.UmbrellaWindow(self.k, self.window, simtrials, f"window_{self.window:.3f}", sim, OP)
        # Equilibrate
        time_average = equili_strict
        UmbrellaEquilibration.naiveTimeAverage(UW, time_average=time_average)
        UW.resetAcceptanceStatistics()
        UW.runUmbrellaTrials(10000)

        # Sampling
        UW.enableLogging()
        UW.resetAcceptanceStatistics()
        UW.runUmbrellaTrials(self.SamplingNumber)
        UW.printUmbrellaVariables()

        logger = hoomd.logging.Logger()
        logger.add(mc, quantities=["type_shapes"])
        hoomd.write.GSD.write(
            state=sim.state,
            filename="force.gsd",
            logger=logger,
            mode="xb",
        )
