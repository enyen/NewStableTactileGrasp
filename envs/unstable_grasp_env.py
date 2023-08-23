import os
import sys
project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)
import cv2
import numpy as np
from gym import spaces
from utils.common import print_info
from scipy.spatial.transform import Rotation
from envs.redmax_torch_env import RedMaxTorchEnv


class UnstableGraspEnv(RedMaxTorchEnv):
    def __init__(self,  render_tactile=False, verbose=False, seed=0):
        # simulation
        asset_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
        model_path = os.path.join(asset_folder, 'unstable_grasp/unstable_grasp.xml')
        self.obs_buf = 0
        super(UnstableGraspEnv, self).__init__(model_path, seed=seed)
        self.sim.viewer_options.camera_lookat = np.array([0., 0., 1])
        self.sim.viewer_options.camera_pos = np.array([3., -1., 1.7])
        self.verbose = verbose
        self.render_tactile = render_tactile

        # observation
        self.tactile_samples, self.tactile_rows, self.tactile_cols = 27, 8, 6
        ndof_obs = self.tactile_samples * self.tactile_rows * self.tactile_cols * 2 * 2
        self.observation_space = spaces.Box(low=np.full(ndof_obs, -float('inf')),
                                            high=np.full(ndof_obs, -float('inf')), dtype=np.float32)
        self.obs_buf = np.zeros(ndof_obs)

        # action
        self.action_space = spaces.Box(low=np.full(2, -1.),
                                       high=np.full(2, 1.), dtype=np.float32)
        self.action_scale = [0.1, 0.003]
        self.hand_bound = 0.1
        self.finger_bound = 0.003

        self.reward_buf = 0.
        self.done_buf = False
        self.info_buf = {}

    def _get_obs(self):
        return self.obs_buf

    def reset(self, seed=-1):
        self.hand_height = 0.166
        self.weight_pos = 0
        # self.domain_rand()
        self.hand_q = 0.
        self.finger_q = 0.
        self.sim.clearBackwardCache()
        self.grasp()
        if seed == -1:
            return self.obs_buf
        else:
            return self.obs_buf, {}

    def domain_rand(self):
        # TODO
        """
        hand height
        weight position, density, size, friction
        contact kn, kt, mu, damping
        tactile kn, kt, mu, damping
        """
        finger_height_range = []
        weight_pos_range = []
        weight_density_range = []
        weight_width_range = []
        weight_mu_range = []
        contact_kn_range = [2e3, 14e3]
        contact_kt_range = [20., 140.]
        contact_mu_range = [0.5, 2.5]
        contact_damping_range = [1e3, 1e3]
        tactile_kn_range = [50, 450]
        tactile_kt_range = [0.2, 2.3]
        tactile_mu_range = [0.5, 2.5]
        tactile_damping_range = [0, 100]

        self.hand_height = self.np_random.uniform(*finger_height_range)
        self.weight_pos = self.np_random.uniform(*weight_pos_range)
        weight_density = self.np_random.uniform(*weight_density_range)
        weight_width = self.np_random.uniform(*weight_width_range)
        weight_mu = self.np_random.uniform(*weight_mu_range)
        contact_kn = self.np_random.uniform(*contact_kn_range)
        contact_kt = self.np_random.uniform(*contact_kt_range)
        contact_mu = self.np_random.uniform(*contact_mu_range)
        contact_damping = self.np_random.uniform(*contact_damping_range)
        tactile_kn = self.np_random.uniform(*tactile_kn_range)
        tactile_kt = self.np_random.uniform(*tactile_kt_range)
        tactile_mu = self.np_random.uniform(*tactile_mu_range)
        tactile_damping = self.np_random.uniform(*tactile_damping_range)

        self.sim.update_body_density('weight', weight_density)
        self.sim.update_body_size('weight', (0.025, weight_width, 0.02))
        self.sim.update_contact_parameters('weight', 'box', mu=weight_mu)
        self.sim.update_contact_parameters('tactile_pad_left', 'box', kn=contact_kn, kt=contact_kt, mu=contact_mu, damping=contact_damping)
        self.sim.update_contact_parameters('tactile_pad_right', 'box', kn=contact_kn, kt=contact_kt, mu=contact_mu, damping=contact_damping)
        self.sim.update_tactile_parameters('tactile_pad_left', kn=tactile_kn, kt=tactile_kt, mu=tactile_mu, damping=tactile_damping)
        self.sim.update_tactile_parameters('tactile_pad_right', kn=tactile_kn, kt=tactile_kt, mu=tactile_mu, damping=tactile_damping)

    def step(self, u):
        action = np.clip(u, -1., 1.) * self.action_scale
        self.hand_q = np.clip(self.hand_q + action[0], -self.hand_bound, self.hand_bound)
        self.finger_q = np.clip(self.finger_q + action[1], -self.finger_bound, self.finger_bound)
        self.grasp()
        truncated = False
        return self.obs_buf, self.reward_buf, self.done_buf, truncated, {'success': self.is_success}

    def grasp(self):
        '''
        each grasp consists of five stage:
        (0) move to pre-grasp
        (1) move to the target grasp position
        (2) close the gripper
        (3) lift and capture the tactile frame
        (4) put down
        (5) open the gripper
        '''
        lift_height = self.hand_height + 0.02
        init_finger = -0.021
        offset_finger = -0.017
        finger_q = self.finger_q + offset_finger

        target_qs = np.array([
            [0.0, self.hand_q, self.hand_height, 0.0, init_finger, init_finger],  # init
            [0.0, self.hand_q, self.hand_height, 0.0, finger_q, finger_q],        # close
            [0.0, self.hand_q, self.hand_height, 0.0, finger_q, finger_q],        # rest
            [0.0, self.hand_q, lift_height, 0.0, finger_q, finger_q],             # up
            [0.0, self.hand_q, lift_height, 0.0, finger_q, finger_q],             # rest
            [0.0, self.hand_q, self.hand_height, 0.0, finger_q, finger_q],        # down
        ])
        num_steps = [80, 20, 200, 40, 200]
        actions, tactile_mask = [], []
        for stage in range(len(num_steps)):  # linear interpolation
            for i in range(num_steps[stage]):
                u = target_qs[stage] + (i + 1) * (target_qs[stage + 1] - target_qs[stage]) / num_steps[stage]
                actions.append(u)
        actions = np.array(actions)
        tactile_mask = np.arange(sum(num_steps)) % 20 == 0

        """
        # qs is the states information of the simulation trajectory
        # qs is (T, ndof_q), each time step, q is a ndof_q-dim vector consisting:
        # qs[t, 0:3]: position of gripper base
        # qs[t, 3]: z-axis rotation of gripper revolute joint
        # qs[t, 4:6]: the positions of two gripper fingers (along x axis)
        # qs[t, 6:9]: the position of the object
        # qs[t, 9:12]: the orientation of the object in rotvec representation
        """
        # simulation
        q_init = np.array([0, self.hand_q, self.hand_height, 0, init_finger, init_finger,
                           0,0,0, 0,0,0,
                           0,self.weight_pos,0, 0,0,0])
        qs, tactiles = self.sim_epi_forward(q_init, actions, tactile_mask)
        self.weight_pos = qs[-1, 13].copy()

        # observation
        self.obs_buf = tactiles.reshape(self.tactile_samples, 2, self.tactile_rows, self.tactile_cols, 3)[..., 0:2].copy()
        self.obs_buf = self.normalize_tactile(self.obs_buf)
        self.obs_buf = self.obs_buf.reshape(-1)

        # reward
        obj_height = qs[-201, 8]
        obj_angle = np.abs(Rotation.from_rotvec(qs[-201, 9:12]).as_euler('xyz', False)[0])
        if obj_angle < 0.001 and obj_height > 0.005:
            if self.verbose:
                print('Success: ', obj_angle)
            self.reward_buf = 100.
            self.done_buf = True
            self.is_success = True
        else:
            if self.verbose:
                print('Failure: ', obj_angle)
            self.reward_buf = -obj_angle * 10.
            self.done_buf = False
            self.is_success = False

    def normalize_tactile(self, tactile_arrays):
        '''
        normalize the shear force field
        input: dimension (T, 2, nrows, ncols, 2)
        output: dimension (T, 2, nrows, ncols, 2)
        '''
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

    def render(self, mode='once'):
        if self.render_tactile:
            tactile_obs = self.get_tactile_obs_array()
            print(tactile_obs.shape)
            tactile_obs = tactile_obs[17:18]
            print(tactile_obs.shape)
            img_tactile_left = self.visualize_tactile(tactile_obs[:, 0:1, ...])
            img_tactile_right = self.visualize_tactile(tactile_obs[:, 1:2, ...])
            img_tactile_left = img_tactile_left.transpose([1, 0, 2])
            img_tactile_right = img_tactile_right.transpose([1, 0, 2])

            cv2.imshow("tactile_left", img_tactile_left)
            cv2.imshow("tactile_right", img_tactile_right)

            print_info('Press [Esc] to continue.')
            cv2.waitKey(0)

        if mode == 'record':
            super().render(mode='record')
        else:
            print_info('Press [Esc] to continue.')
            super().render(mode)

    def get_tactile_obs_array(self):
        tactile_obs = self.obs_buf.reshape(self.tactile_samples, 2, self.tactile_rows, self.tactile_cols, 2)
        return tactile_obs

    def sim_epi_forward(self, q0, actions, tactile_masks):
        self.sim.set_state_init(q0, np.zeros_like(q0))
        self.sim.reset(backward_flag=False)

        qs, tactiles = [], []
        for t in range(actions.shape[0]):
            self.sim.set_u(actions[t])
            self.sim.forward(1, verbose=False, test_derivatives=False)

            qs.append(self.sim.get_q().copy())
            if tactile_masks[t]:
                tactiles.append(self.sim.get_tactile_force_vector().copy())

        qs = np.stack(qs, axis=0)
        tactiles = np.stack(tactiles, axis=0)
        return qs, tactiles
