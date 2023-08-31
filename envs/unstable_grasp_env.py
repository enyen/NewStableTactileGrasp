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
# qs[3]: z-axis rotation of gripper revolute joint
# qs[4:6]: prismatic position of two gripper fingers (along x axis), open(-ve), close(+ve) 
# qs[6:9]: xyz position of the bar
# qs[9:12]: the orientation of the bar in rotvec representation
# qs[12:15]: xyz position of weight
# qs[15:18]: the orientation of weight in rotvec representation
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
        self.tactile_samples, self.tactile_sensors, self.tactile_dim, self.tactile_rows, self.tactile_cols = 4, 2, 2, 8, 6
        ndof_obs = (self.tactile_samples, self.tactile_sensors, self.tactile_dim, self.tactile_rows, self.tactile_cols)
        self.observation_space = gym.spaces.Box(low=np.full(ndof_obs, -float('inf')),
                                                high=np.full(ndof_obs, -float('inf')), dtype=np.float32)
        self.obs_buf = np.zeros(ndof_obs, dtype=np.float32)

        # action
        self.action_space = gym.spaces.Box(low=np.full(2, -1.),
                                           high=np.full(2, 1.), dtype=np.float32)
        self.action_scale = [0.066, 0.002]
        self.hand_bound = 0.1
        self.finger_bound = 0.003

        self.reward_buf = 0.
        self.done_buf = False
        self.info_buf = {}
        self.i_steps = 0

    def reset(self, seed=-1, options=None):
        # randomization
        self.domain_rand()

        # clear
        self.i_steps = 0
        self.hand_q = 0.
        self.finger_q = 0.
        self.sim.clearBackwardCache()

        # init observation
        self.grasp()
        if seed == -1:
            return self.obs_buf
        else:
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
        weight_density_range = [500, 700]
        weight_mu_range = [0.05, 0.15]
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
        self.sim.update_contact_parameters('weight', 'box', mu=weight_mu, kn=5e3, kt=5, damping=1e2)
        self.sim.update_body_size('weight', (0.025, self.weight_width, 0.02))
        # self.sim.update_contact_parameters('tactile_pad_left', 'box', kn=contact_kn, kt=contact_kt, mu=contact_mu, damping=contact_damping)
        # self.sim.update_contact_parameters('tactile_pad_right', 'box', kn=contact_kn, kt=contact_kt, mu=contact_mu, damping=contact_damping)
        # self.sim.update_tactile_parameters('tactile_pad_left', kn=tactile_kn, kt=tactile_kt, mu=tactile_mu, damping=tactile_damping)
        # self.sim.update_tactile_parameters('tactile_pad_right', kn=tactile_kn, kt=tactile_kt, mu=tactile_mu, damping=tactile_damping)

    def step(self, u):
        self.i_steps += 1
        action = np.clip(u, -1., 1.) * self.action_scale
        self.hand_q = np.clip(self.hand_q + action[0], -self.hand_bound, self.hand_bound)
        self.finger_q = np.clip(self.finger_q + action[1], -self.finger_bound, self.finger_bound)
        self.grasp()
        truncated = False
        if self.i_steps == 10:
            truncated = True
            self.done_buf = True
        return self.obs_buf, self.reward_buf, self.done_buf, truncated, {}

    def grasp(self):
        # action
        lift_dist = 0.02
        lift_height = self.hand_height + lift_dist
        init_finger = -0.021
        offset_finger = -0.016
        finger_q = self.finger_q + offset_finger

        target_qs = np.array([
            [0.0, self.hand_q, self.hand_height, 0.0, init_finger, init_finger],  # init
            [0.0, self.hand_q, self.hand_height, 0.0, finger_q, finger_q],        # close
            [0.0, self.hand_q, self.hand_height, 0.0, finger_q, finger_q],        # rest
            [0.0, self.hand_q, lift_height, 0.0, finger_q, finger_q],             # up
            [0.0, self.hand_q, lift_height, 0.0, finger_q, finger_q],             # rest
            [0.0, self.hand_q, self.hand_height, 0.0, finger_q, finger_q],        # down
        ])
        num_steps = [80, 10, 200, 100, 50]
        actions = []
        for stage in range(len(num_steps)):  # linear interpolation
            for i in range(num_steps[stage]):
                u = target_qs[stage] + (i + 1) * (target_qs[stage + 1] - target_qs[stage]) / num_steps[stage]
                actions.append(u)
        actions = np.array(actions)
        # tactile_mask = np.arange(sum(num_steps)) % 20 == 0
        # tactile_mask[-num_steps[-1]:] = False
        tactile_mask = np.zeros(sum(num_steps), dtype=bool)
        tactile_mask[[90, 190, 290, 390]] = True
        q_mask = np.zeros(sum(num_steps), dtype=bool)
        q_mask[-(num_steps[-3] + num_steps[-2] + num_steps[-1]):-num_steps[-1]] = True

        # simulation
        q_init = np.array([0, self.hand_q, self.hand_height, 0, init_finger, init_finger,
                           0,0,0, 0,0,0,
                           0,self.weight_pos,0, 0,0,0])
        qs, tactiles = self.sim_epi_forward(q_init, actions, tactile_mask, q_mask)
        self.weight_pos = np.clip(qs[-1, 13].copy(), -(0.11 - self.weight_width/2 - 0.001),
                                  (0.11 - self.weight_width/2 - 0.001))

        # observation
        obs_buf = rearrange(tactiles, 't (s h w d) -> (t s h w) d',
                            t=self.tactile_samples, s=self.tactile_sensors, h=self.tactile_rows, w=self.tactile_cols, d=3)[..., 0:self.tactile_dim]
        self.obs_buf = rearrange(self.normalize_tactile(obs_buf), '(t s h w) d -> t s d h w',
                                 t=self.tactile_samples, s=self.tactile_sensors, h=self.tactile_rows, w=self.tactile_cols, d=self.tactile_dim)

        # reward
        th_rot = -0.002
        th_drop = -0.002
        box_rot = -np.linalg.norm(qs[:, 9:12], axis=1).max()                # min-0.095
        box_drop = min(0, (qs[0, 2] - qs[0, 8]) - (qs[-1, 2] - qs[-1, 8]))  # min-0.020
        grip_force = -self.finger_bound - self.finger_q                     # min-0.006

        if box_rot > th_rot and box_drop > th_drop:
            # self.reward_buf = (20 +
            #                    1000 * grip_force)
            self.reward_buf = (100 +
                               5000 * grip_force)
            self.done_buf = True
        # elif box_rot < th_rot and box_drop > th_drop:
        #     self.reward_buf = (10 * box_rot +
        #                        1 +
        #                        30 * grip_force)
        #     self.done_buf = False
        else:
            self.reward_buf = (10 * box_rot +
                               20 * box_drop +
                               30 * grip_force)
            self.done_buf = False
        # if self.i_steps == 8:
        #     self.done_buf = True
        # else:
        #     self.done_buf = False

    def normalize_tactile(self, tactile_arrays):
        normalized_tactile_arrays = tactile_arrays.copy()
        lengths = np.linalg.norm(tactile_arrays, axis=-1)
        normalized_tactile_arrays = normalized_tactile_arrays / ((np.max(lengths) + 1e-5) / 30.)
        return normalized_tactile_arrays

    def visualize_tactile(self, tactile_array):
        resolution = 40
        horizontal_space = 20
        vertical_space = 40
        T = len(tactile_array)
        N = tactile_array.shape[1]
        nrows = tactile_array.shape[2]
        ncols = tactile_array.shape[3]

        imgs_tactile = np.zeros(
            (ncols * resolution * N + vertical_space * (N + 1), nrows * resolution * T + horizontal_space * (T + 1), 3),
            dtype=float)

        for timestep in range(T):
            for finger_idx in range(N):
                for row in range(nrows):
                    for col in range(ncols):
                        loc0_x = row * resolution + resolution // 2 + timestep * nrows * resolution + timestep * horizontal_space + horizontal_space
                        loc0_y = col * resolution + resolution // 2 + finger_idx * ncols * resolution + finger_idx * vertical_space + vertical_space
                        loc1_x = loc0_x + tactile_array[timestep][finger_idx][row, col][0] * 1.
                        loc1_y = loc0_y + tactile_array[timestep][finger_idx][row, col][1] * 1.
                        color = (0.0, 1.0, 0.0)
                        cv2.arrowedLine(imgs_tactile, (int(loc0_x), int(loc0_y)), (int(loc1_x), int(loc1_y)), color, 2,
                                        tipLength=0.3)
        return imgs_tactile

    def render_tactile(self, mode='once'):
        if self.render_tactile:
            tactile_obs = self.obs_buf.reshape(self.tactile_samples, 2, self.tactile_rows, self.tactile_cols, 2)
            img_tactile_left = self.visualize_tactile(tactile_obs[-2:-1, 0:1, ...]).transpose([1, 0, 2])
            img_tactile_right = self.visualize_tactile(tactile_obs[-2:-1, 1:2, ...]).transpose([1, 0, 2])
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
        tactiles = np.stack(tactiles, axis=0, dtype=np.float32)
        return qs, tactiles

    def close(self):
        pass
