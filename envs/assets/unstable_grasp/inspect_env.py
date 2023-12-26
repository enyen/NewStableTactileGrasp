import redmax_py as redmax
import numpy as np
from einops import rearrange
import cv2
from matplotlib import pyplot as plt


def visualize_tactile(tactile_array, tactile_resolution=50, shear_force_threshold=0.0001):
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
# sim.viewer_options.camera_lookat = np.array([0., -0.2, 0.5])
# sim.viewer_options.camera_pos = np.array([2., -1., 1])
sim.viewer_options.camera_lookat = np.array([0., 0, 0.5])
sim.viewer_options.camera_pos = np.array([2., 0, 1])
sim.viewer_options.loop = False
sim.viewer_options.infinite = False
sim.viewer_options.speed = 1

qpos_init = sim.get_q_init().copy()
qpos_init[1] = -0
qpos_init[2] = 0.166
qpos_init[3] = -0.02
qpos_init[4] = -0.02
# qpos_init[3] = -0.0185  # 3300, 0.015, 0.1-0.15
# qpos_init[4] = -0.0185
# qpos_init[3] = -0.0135  # 5000, 0.040, 0.09-14
# qpos_init[4] = -0.0135
qpos_init[12] = 0

qpos_end = qpos_init.copy()
qpos_end[2] += 0.02
# sim.update_body_density('load', 5000)
# sim.update_body_size('load', (0.025, 0.04, 0.02))
# force = 0.35
sim.update_body_density('load', 3300)
sim.update_body_size('load', (0.025, 0.015, 0.02))
force = 0.1
# sim.update_body_density('load', 4545)
# sim.update_body_size('load', (0.025, 0.0275, 0.02))
# force = 0.22
sim.update_contact_parameters('load', 'box', mu=0.08, kn=5e3, kt=1e2, damping=1e2)
sim.set_state_init(qpos_init, np.zeros_like(qpos_init))
sim.reset(backward_flag=False)

# up
qs = []
ts = []
for t in range(400):
    u = qpos_init[:5] + (qpos_end[:5] - qpos_init[:5]) * (np.sin(t * np.pi / 400 - np.pi / 2) + 1) / 2
    u[3] = force
    u[4] = force
    sim.set_u(u)
    sim.forward(1, verbose=False, test_derivatives=False)
    qs.append(sim.get_q().copy())
    ts.append(sim.get_tactile_force_vector().copy())
# down
for t in range(50):
    u = qpos_end[:5] + (qpos_init[:5] - qpos_end[:5]) * (np.sin(t * np.pi / 50 - np.pi / 2) + 1) / 2
    u[3] = force
    u[4] = force
    sim.set_u(u)
    sim.forward(1, verbose=False, test_derivatives=False)
    qs.append(sim.get_q().copy())
    ts.append(sim.get_tactile_force_vector().copy())
# wait
# for t in range(50):
#     sim.set_u(qpos_init[:5])
#     sim.forward(1, verbose=False, test_derivatives=False)
#     qs.append(sim.get_q().copy())
#     ts.append(sim.get_tactile_force_vector().copy())

ts = np.stack(ts, axis=0)
qs = np.stack(qs, axis=0)

print('bar angle: ', np.linalg.norm(qs[:, 8:11], axis=-1).max())
print('bar height diff:', min(0, (qs[0, 2] - qs[0, 7]) - (qs[-1, 2] - qs[-1, 7])))
print(qs[0, 12], qs[-1, 12])

# tactile map image
nstep = qs.shape[0]
ts = rearrange(ts, 't (s h w d) -> (t s h w) d', t=nstep, s=1, h=8, w=6, d=3)[..., 0:2]
ts = rearrange(ts, '(t s h w) d -> t s d h w', t=nstep, s=1, h=8, w=6, d=2)
for t in range(nstep):
    img = visualize_tactile(ts[t, 0].transpose([1, 2, 0]))
    cv2.imshow("img", img)
    cv2.waitKey(50)

ts_offset = 0
# plot
fig, ax1 = plt.subplots()
ax1.plot(np.linspace(0, 2, ts.shape[0] - ts_offset), np.linalg.norm(ts, axis=-1).reshape(nstep, -1).mean(axis=1)[ts_offset:] / 0.0006, color='b')
ax1.set_ylabel('Shear Force Magnitude', color='b')
ax1.set_xlabel('Time (s)')

ax2 = ax1.twinx()
# ax2.plot((qs[:, 2] - 0.166) / 0.02)
# ax2.plot((np.sin(np.arange(nstep) * np.pi / nstep - np.pi / 2) + 1) / 2)
ax2.plot(np.linspace(0, 2, ts.shape[0] - ts_offset), qs[ts_offset:, 12], color='g')
ax2.set_ylim(-0.1, 0.1)
ax2.set_ylabel('Load Location (mm)', color='g')
ax2.tick_params(axis='y', labelcolor='g')
plt.show()

sim.replay()
