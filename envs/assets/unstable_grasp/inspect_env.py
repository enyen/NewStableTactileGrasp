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
qpos_init[4] = -0.017
qpos_init[5] = -0.017

qpos_end = qpos_init.copy()
qpos_end[2] += 0.02
sim.set_q_init(qpos_init)
sim.reset(backward_flag=False)

# up
for t in range(200):
    u = qpos_init[:6] + t / 200. * (qpos_end[:6] - qpos_init[:6])
    sim.set_u(u)
    sim.forward(1, verbose=False, test_derivatives=False)
# wait
sim.forward(40, verbose=False, test_derivatives=False)
# down
for t in range(50):
    u = qpos_end[:6] + t / 50. * (qpos_init[:6] - qpos_end[:6])
    sim.set_u(u)
    sim.forward(1, verbose=False, test_derivatives=False)

sim.replay()
