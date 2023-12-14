import redmax_py as redmax
import numpy as np
from einops import rearrange
import cv2


def visualize_tactile(tactile_array, tactile_resolution=50, shear_force_threshold=0.0005):
    resolution = tactile_resolution
    nrows = tactile_array.shape[0]
    ncols = tactile_array.shape[1]

    imgs_tactile = np.zeros((nrows * resolution, ncols * resolution, 3), dtype=float)

    for row in range(nrows):
        for col in range(ncols):
            loc0_x = row * resolution + resolution // 2
            loc0_y = col * resolution + resolution // 2
            loc1_x = loc0_x + tactile_array[row, col][0] / shear_force_threshold * resolution
            loc1_y = loc0_y + tactile_array[row, col][1] / shear_force_threshold * resolution
            color = (1, 0.2, 0.2)
            cv2.arrowedLine(imgs_tactile, (int(loc0_y), int(loc0_x)), (int(loc1_y), int(loc1_x)), color,
                            3, tipLength=0.4)

    return imgs_tactile


sim = redmax.Simulation('./unstable_grasp.xml', verbose=True)
sim.viewer_options.camera_lookat = np.array([0., -0.2, 0.5])
sim.viewer_options.camera_pos = np.array([2., -1., 1])
sim.viewer_options.loop = False
sim.viewer_options.infinite = False
sim.viewer_options.speed = 1

qpos_init = sim.get_q_init().copy()
qpos_init[1] = 0.02
qpos_init[2] = 0.166
qpos_init[3] = -0.015
qpos_init[4] = -0.015
# qpos_init[3] = -0.0185  # 3300, 0.015, 0.1-0.15
# qpos_init[4] = -0.0185
# qpos_init[3] = -0.0135  # 5000, 0.040, 0.09-14
# qpos_init[4] = -0.0135
qpos_init[12] = 0.0

qpos_end = qpos_init.copy()
qpos_end[2] += 0.02

sim.set_state_init(qpos_init, np.zeros_like(qpos_init))
sim.reset(backward_flag=False)
# sim.update_contact_parameters('weight', 'box', mu=0.06, kn=5e3, kt=1e2, damping=1e2)
sim.update_contact_parameters('weight', 'box', mu=0.09, kn=5e3, kt=1e2, damping=1e2)
sim.update_body_density('weight', 4500)
sim.update_body_size('weight', (0.025, 0.03, 0.02))
# sim.update_body_density('weight', 3300)
# sim.update_body_size('weight', (0.025, 0.015, 0.02))
# sim.update_body_density('weight', 5000)
# sim.update_body_size('weight', (0.025, 0.040, 0.02))

# up
nstep = 400
qs = []
ts = []
for t in range(nstep):
    u = qpos_init[:5] + (qpos_end[:5] - qpos_init[:5]) * (np.sin(t * np.pi / nstep - np.pi / 2) + 1) / 2
    sim.set_u(u)
    sim.forward(1, verbose=False, test_derivatives=False)
    qs.append(sim.get_q().copy())
    ts.append(sim.get_tactile_force_vector().copy())
qs = np.stack(qs, axis=0)
ts = np.stack(ts, axis=0)
# print(np.linalg.norm(qs[:, 8:11], axis=-1).max())
# print(min(0, (qs[0, 2] - qs[0, 7]) - (qs[-1, 2] - qs[-1, 7])))

ts = rearrange(ts, 't (s h w d) -> (t s h w) d', t=nstep, s=1, h=8, w=6, d=3)[..., 0:2]
ts = rearrange(ts, '(t s h w) d -> t s d h w', t=nstep, s=1, h=8, w=6, d=2)
for t in range(nstep):
    img = visualize_tactile(ts[t, 0].transpose([1, 2, 0]))
    cv2.imshow("img", img)
    cv2.waitKey(5)

# from matplotlib import pyplot as plt
# fig, ax1 = plt.subplots()
# ax1.plot(np.linalg.norm(ts, axis=-1).reshape(nstep, -1).mean(axis=1))
# ax1.set_ylim(0, 0.0008)
# ax2 = ax1.twinx()
# ax2.plot((qs[:, 2] - 0.166) / 0.02)
# ax2.plot((np.sin(np.arange(nstep) * np.pi / nstep - np.pi / 2) + 1) / 2)
# plt.show()

# wait
# sim.forward(400, verbose=False, test_derivatives=False)

# down
# for t in range(50):
#     u = qpos_end[:6] + t / 50. * (qpos_init[:6] - qpos_end[:6])
#     sim.set_u(u)
#     sim.forward(1, verbose=False, test_derivatives=False)

sim.replay()
