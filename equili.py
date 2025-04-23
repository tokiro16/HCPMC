import hoomd
import os
import hcpmc
path = os.getcwd()
import numpy as np
import sys
import datetime
samplename = path + f'/visualization/{sys.argv[1]}'
factory = hcpmc.initializer.particlefactory.Family423(1.65,2.286)
start_time = datetime.datetime.now()
sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=12345)
sim.create_state_from_gsd(samplename)
mc = factory.get_integrator()
sim.operations.integrator = mc

origin = hcpmc.OrderParameter.OrderParameterAveQlNear(4, 1.8, 12).getOrder(sim.state.get_snapshot())
sim.run(1e4)
ops = []
for i in range(5000):
    sim.run(4)
    orderparameter = hcpmc.OrderParameter.OrderParameterAveQlNear(4, 1.8, 12)
    ops.append(orderparameter.getOrder(sim.state.get_snapshot()))
duration = datetime.datetime.now() - start_time
formatted_duration = str(duration).split(".")[0]
op = np.mean(ops)
print(f"{formatted_duration} equili finished. op target: {op:.4f} NOW try to get equilibrium configuration.")

idx = 1
while idx:
    sim.run(5)
    op_temp = orderparameter.getOrder(sim.state.get_snapshot())
    if abs(op_temp - op)/op < 0.01:
        idx = 0
    if sim.timestep % 1000 == 0:
        duration = datetime.datetime.now() - start_time
        formatted_duration = str(duration).split(".")[0]
        print(f"{formatted_duration} trying to get equilibrium configuration. NOW op: {op_temp:.4f}")
logger = hoomd.logging.Logger()
logger.add(mc, quantities=["type_shapes"])
hoomd.write.GSD.write(
    state=sim.state,
    filename=path + f'/visualization/{origin:.4f}_FL.gsd',
    logger=logger,
    mode="xb",
    )
duration = datetime.datetime.now() - start_time
formatted_duration = str(duration).split(".")[0]
print(f"{formatted_duration} finished.")