from flow import FlowProject
import numpy as np
import datetime
import hoomd


class Densest(FlowProject):
    pass


@Densest.label
def finish(job):
    return job.isfile("trial.gsd")


@Densest.post(finish)
@Densest.operation(directives={"memory": "1g", "walltime": 24})
def compress(job):
    with job:
        start_time = datetime.datetime.now()

        temp = job.path + "/../../dense_temp.gsd"
        sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=job.sp.seed)
        sim.create_state_from_gsd(temp)
        mc = hoomd.hpmc.integrate.ConvexSpheropolygon(default_d=0.1, default_a=0.1)
        mc.shape["S0"] = dict(vertices=job.sp.mc_vertices, sweep_radius=job.sp.mc_radius)
        sim.operations.integrator = mc
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
        logger.add(mc, quantities=["type_shapes"])
        gsd_writer = hoomd.write.GSD(
            filename="compressing.gsd",
            trigger=hoomd.trigger.Periodic(40000),
            logger=logger,
            mode="xb",
        )
        sim.operations.writers.append(gsd_writer)
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
        gsd_writer.flush()
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
            now_rho = job.sp.particle_per_cell / (sim.state.box.volume)
            print(
                f"{formatted_duration}  Generating {i + 1}/{len(lgP_target)}. Now rho: {now_rho:.4f}. ",
                flush=True,
            )
            sim.operations.updaters.remove(Boxmc)
            gsd_writer.flush()

        # Initialize GSD file writer
        phi = job.sp.particle_per_cell / sim.state.box.volume
        job.doc["phi"] = phi
        hoomd.write.GSD.write(
            state=sim.state,
            mode="wb",
            filename="trial.gsd",
            filter=hoomd.filter.All(),
            logger=logger,
        )


if __name__ == "__main__":
    Densest().main()
