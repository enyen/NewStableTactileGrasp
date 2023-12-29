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
# qs[3:5]: prismatic position of two gripper fingers (along x axis), open(-ve), close(+ve) 
# qs[5:8]: xyz position of the bar
# qs[8:11]: the orientation of the bar in rotvec representation
# qs[11:14]: xyz position of load
# qs[14:17]: the orientation of load in rotvec representation
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
        self.hand_bound = [-0.102, 0.102]
        self.finger_bound = [0.1, 0.35]
        self.tactile_means = np.array([[1.07513879e-04, -1.76299886e-07]])
        self.tactile_stds = np.array([[0.00014761, 0.00014823]])
        self.tactile_noise = 0

        self.reward_buf = 0.
        self.done_buf = False
        self.info_buf = {}
        self.i_steps = 0
        self.acc_reward = 0
        self.hand_q = 0.
        self.finger_force = (self.finger_bound[1] - self.finger_bound[0]) / 2
        _ = self.reset()

    def reset(self, seed=None, options=None):
        # randomization
        self.domain_rand()

        # clear
        self.i_steps = 0
        self.acc_reward = 0
        self.hand_q = 0.
        self.finger_force = (self.finger_bound[1] - self.finger_bound[0]) / 2
        self.sim.clearBackwardCache()

        # init observation
        self.grasp()
        return self.obs_buf, {}

    def domain_rand(self):
        """
        hand height
        load position, density, size, friction
        """
        finger_height_range = [0.163, 0.169]
        load_mu_range = [0.11, 0.21]
        load_pos_range = [-0.089, 0.089]
        load_width_range = [0.015, 0.040]
        load_density_range = [3300, 5000]

        self.hand_height = self.np_random.uniform(*finger_height_range)
        load_mu = self.np_random.uniform(*load_mu_range)
        self.load_pos = self.np_random.uniform(*load_pos_range)
        self.load_width = self.np_random.uniform(*load_width_range)
        load_density = self.np_random.uniform(*load_density_range)

        self.sim.update_body_color('load', tuple(self.np_random.uniform(0, 1, 3)))
        self.sim.update_body_density('load', load_density)
        self.sim.update_body_size('load', (0.025, self.load_width, 0.02))
        self.sim.update_contact_parameters('load', 'box', mu=load_mu, kn=5e3, kt=1e2, damping=1e2)
        self.load_weight = ((self.load_width * load_density - load_width_range[0] * load_density_range[0]) /
                            (load_width_range[1] * load_density_range[1] - load_width_range[0] * load_density_range[0]))

    def step(self, u):
        self.i_steps += 1
        action = np.clip(u, -1., 1.) * [(self.hand_bound[1] - self.hand_bound[0]) / 2,
                                        (self.finger_bound[1] - self.finger_bound[0]) / 2]
        self.hand_q = np.clip(self.hand_q + action[0], self.hand_bound[0], self.hand_bound[1])
        self.finger_force = np.clip(self.finger_force + action[1], self.finger_bound[0], self.finger_bound[1])
        self.grasp()
        truncated = False
        info = {}
        if self.i_steps == 10 and not self.done_buf:
            truncated = True
            info["TimeLimit.truncated"] = True
        self.acc_reward += self.reward_buf
        return self.obs_buf, self.reward_buf, self.done_buf, truncated, info

    def grasp(self):
        # action
        finger_init = -0.02
        lift_dist = 0.02

        target_qs = np.array([
            [0.0, self.hand_q, self.hand_height, self.finger_force, self.finger_force],              # close
            [0.0, self.hand_q, self.hand_height + lift_dist, self.finger_force, self.finger_force],  # up
            [0.0, self.hand_q, self.hand_height, self.finger_force, self.finger_force],              # down
        ])
        num_steps = [400, 50]
        actions = []
        for stage in range(len(num_steps)):  # sine interpolation
            for i in range(num_steps[stage]):
                u = (target_qs[stage] + (target_qs[stage + 1] - target_qs[stage]) *
                     (np.sin((i + 1) * np.pi / num_steps[stage] - np.pi / 2) + 1) / 2)
                actions.append(u)
        actions = np.array(actions)

        # simulation
        q_init = np.array([0, self.hand_q, self.hand_height, finger_init, finger_init,
                           0,0,0, 0,0,0,
                           0,self.load_pos,0, 0,0,0])
        load_pos, box_orien, gripper_height, box_height, tactiles = (
            self.sim_epi_forward(q_init, actions, num_steps[0], self.tactile_samples))
        self.load_pos = np.clip(load_pos[-1], -(0.11 - self.load_width / 2), (0.11 - self.load_width / 2))

        # observation
        # obs_buf = tactiles - tactiles[0:1]
        obs_buf = rearrange(tactiles, 't (s h w d) -> (t s h w) d', t=self.tactile_samples,  s=self.tactile_sensors,
                            h=self.tactile_rows, w=self.tactile_cols, d=3)[..., 0:self.tactile_dim]
        obs_buf = self.normalize_tactile(obs_buf)
        self.obs_buf = rearrange(obs_buf, '(t s h w) d -> t s d h w', t=self.tactile_samples, s=self.tactile_sensors,
                                 h=self.tactile_rows, w=self.tactile_cols, d=self.tactile_dim)

        # reward
        th_rot = 0.22   # 0.02rad, 1.15degree
        th_drop = 0.15  # 3mm
        box_rot = min(np.linalg.norm(box_orien, axis=1).max(), 0.091) / 0.091
        box_drop = max(0, min(lift_dist, (gripper_height[-1] - gripper_height[0]) - (box_height[-1] - box_height[0]))) / lift_dist
        force_diff = (self.finger_force - self.finger_bound[0]) / (self.finger_bound[1] - self.finger_bound[0]) - self.load_weight

        self.done_buf = False
        self.reward_buf = max(box_rot, box_drop) * -1.
        if box_rot < th_rot and box_drop < th_drop:
            self.done_buf = True
            self.reward_buf = min(1. / (max(0, force_diff) + 1e-4), 10.)

    def normalize_tactile(self, tactile_arrays):
        # normalized_tactile_arrays = tactile_arrays.copy()
        # lengths = np.linalg.norm(tactile_arrays, axis=-1)
        # normalized_tactile_arrays = normalized_tactile_arrays / ((np.max(lengths) + 1e-5) / 30.)
        # return normalized_tactile_arrays
        tactile_arrays = (tactile_arrays - self.tactile_means) / self.tactile_stds
        if self.tactile_noise > 0:
            tactile_arrays = (tactile_arrays +
                              self.np_random.uniform(-self.tactile_noise, self.tactile_noise, tactile_arrays.shape))
        return tactile_arrays

    def visualize_tactile(self, tactile_array, tactile_resolution=50, shear_force_threshold=0.0001):
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

    def render_tactile(self):
        obs_buf = rearrange(self.obs_buf, 't s d h w -> (t s h w) d')
        obs_buf = obs_buf * self.tactile_stds + self.tactile_means
        obs_buf = rearrange(obs_buf, '(t s h w) d -> t s d h w', t=self.tactile_samples,
                            s=self.tactile_sensors, h=self.tactile_rows, w=self.tactile_cols)
        for t in range(self.obs_buf.shape[0]):
            img = self.visualize_tactile(obs_buf[t, 0].transpose([1, 2, 0]))
            cv2.imshow("img", img)
            cv2.waitKey(200)
        self.render()

    def render(self):
        if self.render_style == 'show':
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

    def sim_epi_forward(self, q0, actions, num_steps, num_tactile):
        t_tactile = np.linspace(8, num_steps, num_tactile).round().astype(int).tolist()
        self.sim.set_state_init(q0, np.zeros_like(q0))
        self.sim.reset(backward_flag=False)

        qs, tactiles = [], []
        for t in range(actions.shape[0]):
            self.sim.set_u(actions[t])
            self.sim.forward(1, verbose=False, test_derivatives=False)

            qs.append(self.sim.get_q().copy())
            if t in t_tactile:
                tactiles.append(self.sim.get_tactile_force_vector().copy())

        qs = np.stack(qs, axis=0)
        load_pos = qs[:, 12]
        box_orien = qs[:num_steps, 8:11]
        gripper_height = qs[:num_steps, 2]
        box_height = qs[:num_steps, 7]
        tactiles = np.stack(tactiles, axis=0, dtype=np.float32)
        return load_pos, box_orien, gripper_height, box_height, tactiles

    def close(self):
        pass

    def data_stat(self):
        from rich.progress import track
        # disable normalization
        self.tactile_means = 0
        self.tactile_stds = 1

        # collect values
        data = []
        self.reset()
        for _ in track(range(256), "Collecting..."):
            obs, _, termi, trunc, _ = self.step(self.np_random.uniform(-1, 1, 2))
            data.append(obs)
            if termi or trunc:
                self.reset()

        data = np.concatenate(data, axis=0)
        data = rearrange(data, 't s c h w -> (t s h w) c')
        print('mean:', data.mean(axis=0), 'std:', data.std(axis=0))
