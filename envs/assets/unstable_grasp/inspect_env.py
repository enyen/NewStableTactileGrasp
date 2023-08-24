import redmax_py as redmax
import numpy as np

sim = redmax.Simulation('./unstable_grasp.xml', verbose=True)
sim.viewer_options.loop = False
sim.viewer_options.infinite = False
sim.viewer_options.speed = 1

qpos_init = sim.get_q_init().copy()
qpos_init[1] = 0.003
qpos_init[2] = 0.166
qpos_init[4] = -0.016
qpos_init[5] = -0.016
qpos_init[13] = 0

qpos_end = qpos_init.copy()
qpos_end[2] += 0.02

sim.set_state_init(qpos_init, np.zeros_like(qpos_init))
sim.reset(backward_flag=False)
print(sim.get_q())

# up
for t in range(200):
    u = qpos_init[:6] + t / 200. * (qpos_end[:6] - qpos_init[:6])
    sim.set_u(u)
    sim.forward(1, verbose=False, test_derivatives=False)
# wait
sim.forward(100, verbose=False, test_derivatives=False)
print(sim.get_q())
print(np.linalg.norm(sim.get_q()[9:12]))
# down
for t in range(50):
    u = qpos_end[:6] + t / 50. * (qpos_init[:6] - qpos_end[:6])
    sim.set_u(u)
    sim.forward(1, verbose=False, test_derivatives=False)

sim.replay()
