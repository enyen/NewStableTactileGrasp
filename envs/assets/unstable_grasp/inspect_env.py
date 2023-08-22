import redmax_py as redmax
import numpy as np

sim = redmax.Simulation('./unstable_grasp.xml', verbose=True)
sim.viewer_options.loop = False
sim.viewer_options.infinite = False
sim.viewer_options.speed = 1

qpos_init = sim.get_q_init().copy()
print("q_init: ", qpos_init.shape)
qpos_init[1] = -0.05
qpos_init[2] = 0.166
qpos_init[4] = -0.02
qpos_init[5] = -0.02
qpos_end = qpos_init.copy()
qpos_end[2] = 0.21
sim.set_q_init(qpos_init)
sim.reset(backward_flag=False)
for t in range(500):
    u = qpos_init[:6] + t / 500. * (qpos_end[:6] - qpos_init[:6])
    # u[4] = 0.04
    # u[5] = 0.04
    sim.set_u(u)
    sim.forward(1, verbose=False, test_derivatives=False)
# sim.forward(1000, verbose=False, test_derivatives=False)
sim.replay()
