from os import path, system
import cv2
import numpy as np
import redmax_py as redmax
import gymnasium as gym
from gymnasium.utils import seeding
from einops import rearrange

"""
dof definition:
# qs[0:3]: xyz position of gripper base
# "removed" qs[3]: z-axis rotation of gripper revolute joint
# qs[3:5]: prismatic position of two gripper fingers (along x axis), open(-ve), close(+ve) 
# qs[5:8]: xyz position of the bar
# qs[8:11]: the orientation of the bar in rotvec representation
# qs[11:14]: xyz position of weight
# qs[14:17]: the orientation of weight in rotvec representation
"""


class UnstableGraspEnv(gym.Env):
    metadata = {}

    def __init__(self, render_style='loop', seed=None):
        super().__init__()
        # simulation
        model_path = path.join(path.join(path.dirname(path.abspath(__file__)), 'assets'),
                               'unstable_grasp/unstable_grasp.xml')
        self.sim = redmax.Simulation(model_path)
        self.sim.viewer_options.camera_lookat = np.array([0., 0., 1])
        self.sim.viewer_options.camera_pos = np.array([3., -1., 1.7])
        self.render_mode = 'rgb'
        self.render_style = render_style
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        # observation
        self.tactile_samples, self.tactile_sensors, self.tactile_dim, self.tactile_rows, self.tactile_cols = 11, 1, 2, 8, 6
        ndof_obs = (self.tactile_samples, self.tactile_sensors, self.tactile_dim, self.tactile_rows, self.tactile_cols)
        self.observation_space = gym.spaces.Box(low=np.full(ndof_obs, -10),
                                                high=np.full(ndof_obs, 10), dtype=np.float32)
        self.obs_buf = np.zeros(ndof_obs, dtype=np.float32)

        # action
        self.action_space = gym.spaces.Box(low=np.full(2, -1.),
                                           high=np.full(2, 1.), dtype=np.float32)
        self.hand_bound = 0.1
        self.finger_bound = 0.0025
        # self.tactile_noise = 1e-6
        self.tactile_noise = 1e-5

        self.reward_buf = 0.
        self.done_buf = False
        self.info_buf = {}
        self.i_steps = 0
        self.acc_reward = 0
        self.hand_q = 0.
        self.finger_q = 0.
        _ = self.reset()

    def reset(self, seed=None, options=None):
        # randomization
        self.domain_rand()

        # clear
        self.i_steps = 0
        self.acc_reward = 0
        self.hand_q = 0.
        self.finger_q = 0.
        self.sim.clearBackwardCache()

        # init observation
        self.grasp()
        return self.obs_buf, {}

    def domain_rand(self):
        # TODO: set ranges
        """
        hand height
        weight position, density, size, friction
        contact kn, kt, mu, damping
        tactile kn, kt, mu, damping
        """
        finger_height_range = [0.163, 0.169]
        weight_pos_range = [-0.089, 0.089]
        weight_width_range = [0.015, 0.040]
        weight_density_range = [3300, 5000]
        weight_mu_range = [0.06, 0.1]
        # contact_kn_range = [2e3, 14e3]
        # contact_kt_range = [20., 140.]
        # contact_mu_range = [0.5, 2.5]
        # contact_damping_range = [1e3, 1e3]
        # tactile_kn_range = [50, 450]
        # tactile_kt_range = [0.2, 2.3]
        # tactile_mu_range = [0.5, 2.5]
        # tactile_damping_range = [0, 100]

        self.hand_height = self.np_random.uniform(*finger_height_range)
        self.weight_pos = self.np_random.uniform(*weight_pos_range)
        self.weight_width = self.np_random.uniform(*weight_width_range)
        weight_density = self.np_random.uniform(*weight_density_range)
        weight_mu = self.np_random.uniform(*weight_mu_range)
        # contact_kn = self.np_random.uniform(*contact_kn_range)
        # contact_kt = self.np_random.uniform(*contact_kt_range)
        # contact_mu = self.np_random.uniform(*contact_mu_range)
        # contact_damping = self.np_random.uniform(*contact_damping_range)
        # tactile_kn = self.np_random.uniform(*tactile_kn_range)
        # tactile_kt = self.np_random.uniform(*tactile_kt_range)
        # tactile_mu = self.np_random.uniform(*tactile_mu_range)
        # tactile_damping = self.np_random.uniform(*tactile_damping_range)

        self.sim.update_body_color('weight', tuple(self.np_random.uniform(0, 1, 3)))
        self.sim.update_body_density('weight', weight_density)
        self.sim.update_contact_parameters('weight', 'box', mu=weight_mu, kn=5e3, kt=1e2, damping=1e2)
        self.sim.update_body_size('weight', (0.025, self.weight_width, 0.02))
        self.weight_weight = 0.025 * self.weight_width * 0.02 * weight_density
        # self.sim.update_contact_parameters('tactile_pad_left', 'box', kn=contact_kn, kt=contact_kt, mu=contact_mu, damping=contact_damping)
        # self.sim.update_contact_parameters('tactile_pad_right', 'box', kn=contact_kn, kt=contact_kt, mu=contact_mu, damping=contact_damping)
        # self.sim.update_tactile_parameters('tactile_pad_left', kn=tactile_kn, kt=tactile_kt, mu=tactile_mu, damping=tactile_damping)
        # self.sim.update_tactile_parameters('tactile_pad_right', kn=tactile_kn, kt=tactile_kt, mu=tactile_mu, damping=tactile_damping)

    def step(self, u):
        self.i_steps += 1
        action = np.clip(u, -1., 1.) * 1.0 * [self.hand_bound, self.finger_bound]
        self.hand_q = np.clip(self.hand_q + action[0], -self.hand_bound, self.hand_bound)
        self.finger_q = np.clip(self.finger_q + action[1], -self.finger_bound, self.finger_bound)
        self.grasp()
        truncated = False
        info = {}
        if self.i_steps == 6 and not self.done_buf:
            truncated = True
            info["TimeLimit.truncated"] = True
        self.acc_reward += self.reward_buf
        return self.obs_buf, self.reward_buf, self.done_buf, truncated, info

    def grasp(self):
        # action
        lift_dist = 0.02
        lift_height = self.hand_height + lift_dist
        init_finger = -0.021
        offset_finger = -0.016
        finger_q = self.finger_q + offset_finger

        target_qs = np.array([
            [0.0, self.hand_q, self.hand_height, init_finger, init_finger],  # init
            [0.0, self.hand_q, self.hand_height, finger_q, finger_q],        # close
            [0.0, self.hand_q, self.hand_height, finger_q, finger_q],        # rest
            [0.0, self.hand_q, lift_height, finger_q, finger_q],             # up
            [0.0, self.hand_q, lift_height, finger_q, finger_q],             # rest
            [0.0, self.hand_q, self.hand_height, finger_q, finger_q],        # down
        ])
        num_steps = [80, 10, 300, 200, 50]
        # num_steps = [80, 10, np.random.randint(290, 330), np.random.randint(190, 210), 50]
        actions = []
        for stage in range(len(num_steps)):  # linear interpolation
            for i in range(num_steps[stage]):
                u = target_qs[stage] + (i + 1) * (target_qs[stage + 1] - target_qs[stage]) / num_steps[stage]
                actions.append(u)
        actions = np.array(actions)
        tactile_mask = np.zeros(sum(num_steps), dtype=bool)
        tactile_masks_ = (num_steps[0] + num_steps[1] +
                          # np.linspace(np.random.randint(5), num_steps[2] + num_steps[3] - 1 - np.random.randint(5),
                          np.linspace(0, num_steps[2] + num_steps[3] - 1,
                                      self.tactile_samples).round().astype(int)).tolist()
        tactile_mask[tactile_masks_] = True
        q_mask = np.zeros(sum(num_steps), dtype=bool)
        q_mask[-(num_steps[-3] + num_steps[-2] + num_steps[-1]):-num_steps[-1]] = True

        # simulation
        q_init = np.array([0, self.hand_q, self.hand_height, init_finger, init_finger,
                           0,0,0, 0,0,0,
                           0,self.weight_pos,0, 0,0,0])
        weight_pos, box_orien, gripper_height, box_height, tactiles = (
            self.sim_epi_forward(q_init, actions, tactile_mask, q_mask))
        self.weight_pos = np.clip(weight_pos[-1], -(0.11 - self.weight_width / 2), (0.11 - self.weight_width / 2))

        # observation
        obs_buf = tactiles - tactiles[0:1]
        obs_buf = rearrange(obs_buf, 't (s h w d) -> (t s h w) d', t=self.tactile_samples,  s=self.tactile_sensors,
                            h=self.tactile_rows, w=self.tactile_cols, d=3)[..., 0:self.tactile_dim]
        obs_buf = self.normalize_tactile(obs_buf, self.tactile_noise)
        self.obs_buf = rearrange(obs_buf, '(t s h w) d -> t s d h w', t=self.tactile_samples, s=self.tactile_sensors,
                                 h=self.tactile_rows, w=self.tactile_cols, d=self.tactile_dim)

        # reward
        th_rot = -0.005
        th_drop = -0.002
        box_rot = -min(np.linalg.norm(box_orien, axis=1).max(), 0.092)                # min-0.092
        box_drop = -max(0, min(0.02, (gripper_height[-1] - gripper_height[0]) - (box_height[-1] - box_height[0])))  # min-0.020
        grip_force = -self.finger_bound - self.finger_q                     # min-0.005

        if box_rot > th_rot and box_drop > th_drop:
            self.done_buf = True
            self.reward_buf = (70 +
                               12000 * grip_force)  # [10:70]
        else:
            self.done_buf = False
            self.reward_buf = (22 * box_rot +
                               100 * box_drop +
                               0 * grip_force)  # [(-2-2-0):0]

    def normalize_tactile(self, tactile_arrays, noise):
        # normalized_tactile_arrays = tactile_arrays.copy()
        # lengths = np.linalg.norm(tactile_arrays, axis=-1)
        # normalized_tactile_arrays = normalized_tactile_arrays / ((np.max(lengths) + 1e-5) / 30.)
        # return normalized_tactile_arrays
        tactile_arrays = (tactile_arrays - np.array([1.21156176e-04, 1.00238935e-06])) / np.array([0.00013228, 0.00013304])
        tactile_arrays = tactile_arrays + self.np_random.uniform(-noise, noise, tactile_arrays.shape)
        return tactile_arrays

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
                color = (255, 0, 0)
                cv2.arrowedLine(imgs_tactile, (int(loc0_y), int(loc0_x)), (int(loc1_y), int(loc1_x)), color,
                                2, tipLength=0.4)

        return imgs_tactile

    def render_tactile(self):
        tactile_obs = self.obs_buf.reshape(self.tactile_samples, self.tactile_sensors, self.tactile_rows, self.tactile_cols, self.tactile_dim)
        img_tactile_left = self.visualize_tactile(tactile_obs[-2:-1, 0:1, ...].transpose([1, 2, 0]))
        img_tactile_right = self.visualize_tactile(tactile_obs[-2:-1, 1:2, ...].transpose([1, 2, 0]))
        cv2.imshow("tactile_left", img_tactile_left)
        cv2.imshow("tactile_right", img_tactile_right)
        cv2.waitKey(1)
        print('\033[96m', 'Looping... Press [Esc] to continue.', '\033[0m')
        super().render('loop')

    def render(self):
        if self.render_style == 'loop':
            self.sim.viewer_options.loop = True
            self.sim.viewer_options.infinite = True
            self.sim.viewer_options.speed = 1.
            self.sim.replay()
        elif self.render_style == 'record':
            self.sim.viewer_options.loop = False
            self.sim.viewer_options.infinite = False
            self.sim.viewer_options.speed = 1.
            self.sim.viewer_options.record = True
            self.sim.viewer_options.record_folder = './storage'
            self.sim.replay()
            system("ffmpeg -i ./storage/%d.png -vf palettegen ./storage/palette.png -hide_banner -loglevel error")
            system("ffmpeg -framerate 30 -i ./storage/%d.png -i ./storage/palette.png -lavfi paletteuse ./storage/{}.gif -hide_banner -loglevel error".format(self.i_steps))
            system("rm ./storage/*.png")

    def sim_epi_forward(self, q0, actions, tactile_masks, q_mask):
        self.sim.set_state_init(q0, np.zeros_like(q0))
        self.sim.reset(backward_flag=False)

        qs, tactiles = [], []
        for t in range(actions.shape[0]):
            self.sim.set_u(actions[t])
            self.sim.forward(1, verbose=False, test_derivatives=False)

            if q_mask[t]:
                qs.append(self.sim.get_q().copy())
            if tactile_masks[t]:
                tactiles.append(self.sim.get_tactile_force_vector().copy())

        qs = np.stack(qs, axis=0)
        weight_pos = qs[:, 12]
        box_orien = qs[:, 8:11]
        gripper_height = qs[:, 2]
        box_height = qs[:, 7]
        tactiles = np.stack(tactiles, axis=0, dtype=np.float32)
        return weight_pos, box_orien, gripper_height, box_height, tactiles

    def close(self):
        pass

    def data_stat(self):
        from rich.progress import track
        data = []
        for _ in track(range(200), "Collecting..."):
            self.reset()
            obs, _, termi, trunc, _ = self.step(self.np_random.uniform(-1, 1, 2))
            data.append(obs)
            if termi or trunc:
                self.reset()

        data = np.concatenate(data, axis=0)
        data = rearrange(data, 't s c h w -> (t s h w) c')
        print('mean:', data.mean(axis=0), 'std:', data.std(axis=0))
