import hoomd
import numpy as np
import datetime

class Densest:
    def __init__(self, samplename, particlefactory):
        self.sample = samplename
        self.factory = particlefactory
        self.packing_fraction = None

    def calculate(self, seed = 12345):
        start_time = datetime.datetime.now()
        sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=seed)
        sim.create_state_from_gsd(self.sample)
        mc = self.factory.get_integrator()
        sim.operations.integrator = mc
        sim.run(0)
        if mc.overlaps != 0:
            print("Overlaps found in the initial state")
            return 0
        tune = hoomd.hpmc.tune.MoveSize.scale_solver(
            moves=["a", "d"],
            target=0.1,
            trigger=hoomd.trigger.Periodic(100),
            max_translation_move=0.01,
            max_rotation_move=0.01,
        )
        sim.operations.tuners.append(tune)
        lgP_target = np.linspace(0, 5, 101)
        logger = hoomd.logging.Logger()
        try:
            logger.add(mc, quantities=["type_shapes"])
        except:
            pass
        # gsd_writer = hoomd.write.GSD(
        #     filename="compressing.gsd",
        #     trigger=hoomd.trigger.Periodic(40000),
        #     logger=logger,
        #     mode="xb",
        # )
        # sim.operations.writers.append(gsd_writer)
        sim.run(0, True)
        duration = datetime.datetime.now() - start_time
        formatted_duration = str(duration).split(".")[0]
        print(
            f"{formatted_duration}  Starting compress...Initializing... ",
            flush=True,
        )
        # Boxmc
        Boxmc = hoomd.hpmc.update.BoxMC(trigger=hoomd.trigger.Periodic(1), P=-1)
        Boxmc.aspect.update({"delta": 0.01, "weight": 2.0})
        Boxmc.shear.update({"weight": 1.0, "delta": (0.001, 0.001, 0.001)})
        sim.operations.updaters.append(Boxmc)
        sim.run(40000)
        sim.operations.updaters.remove(Boxmc)

        for i in range(len(lgP_target)):
            # Boxmc
            Boxmc = hoomd.hpmc.update.BoxMC(trigger=hoomd.trigger.Periodic(1), P=10 ** lgP_target[i])
            Boxmc.volume.update({"weight": 1.0, "delta": 0.001 * sim.state.box.volume})
            Boxmc.aspect.update({"delta": 0.001, "weight": 1.0})
            Boxmc.shear.update({"weight": 1.0, "delta": (0.001, 0.001, 0.001)})
            sim.operations.updaters.append(Boxmc)

            sim.run(40000)
            duration = datetime.datetime.now() - start_time
            formatted_duration = str(duration).split(".")[0]
            now_rho = sim.state.N_particles / (sim.state.box.volume)
            print(
                f"{formatted_duration}  Generating {i + 1}/{len(lgP_target)}. Now rho: {now_rho:.4f}. ",
                flush=True,
            )
            sim.operations.updaters.remove(Boxmc)

            phi = sim.state.N_particles / sim.state.box.volume
            self.packing_fraction = phi
            hoomd.write.GSD.write(
                state=sim.state,
                mode="wb",
                filename="trial.gsd",
                filter=hoomd.filter.All(),
                logger=logger,
            )