import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(project_base_dir)

import redmax_py as redmax
import numpy as np
from utils import math
from envs.redmax_torch_env import RedMaxTorchEnv
from utils.common import *
from gym import spaces
import torch
import cv2
from copy import deepcopy
from scipy.spatial.transform import Rotation


class UnstableGraspEnv(RedMaxTorchEnv):
    def __init__(self,  render_tactile=False, verbose=False, seed=0):
        asset_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
        model_path = os.path.join(asset_folder, 'unstable_grasp/unstable_grasp.xml')

        self.verbose = verbose
        self.render_tactile = render_tactile

        self.tactile_rows = 13
        self.tactile_cols = 10

        self.ndof_obs = self.tactile_rows * self.tactile_cols * 2 * 2
        self.obs_shape = (self.tactile_rows * self.tactile_cols * 2 * 2,)

        self.tactile_samples = 1
        self.obs_buf = np.zeros(self.obs_shape)
        self.reward_buf = 0.
        self.done_buf = False
        self.info_buf = {}

        super(UnstableGraspEnv, self).__init__(model_path, seed=seed)

        self.sim.viewer_options.camera_lookat = np.array([0., 0., 1])
        self.sim.viewer_options.camera_pos = np.array([3., -1., 1.7])

        self.ndof_u = 1
        self.action_space = spaces.Box(low=np.full(self.ndof_u, -1.), high=np.full(self.ndof_u, 1.), dtype=np.float32)
        self.action_scale = 0.055
        self.grasp_position_bound = 0.11

    def _get_obs(self):
        return self.obs_buf

    def reset(self, seed=-1):
        """
        number of block: 0 - 26
        com location: 4 - 22
        com std: 3 - 8
        com friction: 5 - 15
        total density: 3000 - 7000
        :param seed:
        :return:
        """
        self.num_blocks = 27
        self.com_blk = self.np_random.uniform(4, self.num_blocks - 4, 1)
        self.std_blk = self.np_random.uniform(4, 12, 1)
        self.sum_blk = self.np_random.uniform(3000, 7000, 1)
        self.fri_blk = self.np_random.uniform(5, 5, 1)
        self.update_density()

        self.sim.clearBackwardCache()
        self.grasp_position = 0.
        self.grasp()
        obs = self.obs_buf
        if seed == -1:
            return obs
        else:
            return obs, {}

    def update_density(self):
        den_blk = np.exp(-np.power(np.arange(self.num_blocks) - self.com_blk, 2) / self.std_blk) + 0.05
        den_blk *= self.sum_blk / den_blk.sum()
        # if self.verbose:
        #     print('Density distribution:', den_blk)
        #     print('Total density:', den_blk.sum())
        for idx in range(self.num_blocks):
            box_name = "box_{}".format(idx)
            self.sim.update_body_density(box_name, den_blk[idx])
            color_light = np.array([0.8, 0.8, 0.8])
            color_heavy = np.array([0.7, 0.3, 0.01])
            color = den_blk[idx] / den_blk.max() * (color_heavy - color_light) + color_light
            self.sim.update_body_color(box_name, color)
        self.den_blk = deepcopy(den_blk.tolist())

    def slide_com(self, angle):
        """
        simple linear model proportional to rotation angle
        :param angle: rotation angle
        :return: change in com, update density
        """
        self.com_blk = np.clip(self.com_blk - angle * self.fri_blk,
                               4, self.num_blocks - 4)
        self.update_density()

    def step(self, u):
        action = np.clip(u, -1., 1.)
        action_unnorm = action * self.action_scale
        self.grasp_position = np.clip(self.grasp_position + action_unnorm[0], -self.grasp_position_bound,
                                         self.grasp_position_bound)

        self.grasp()
        reward, done = self.reward_buf, self.done_buf

        obs = self.obs_buf
        truncated = False
        return obs, reward, done, truncated, {'success': self.is_success}

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

        grasp_height = 0.2029862
        init_height = grasp_height + 0.03
        lift_height = grasp_height + 0.03
        init_finger = -0.03
        grasp_finger = -0.008
        # grasp_finger = -0.0165

        target_qs = np.array([
            [0.0, self.grasp_position, init_height, 0.0, init_finger, init_finger],     # init
            [0.0, self.grasp_position, grasp_height, 0.0, init_finger, init_finger],    # down
            [0.0, self.grasp_position, grasp_height, 0.0, init_finger, init_finger],    # rest
            [0.0, self.grasp_position, grasp_height, 0.0, grasp_finger, grasp_finger],  # close
            [0.0, self.grasp_position, grasp_height, 0.0, grasp_finger, grasp_finger],  # rest
            [0.0, self.grasp_position, lift_height, 0.0, grasp_finger, grasp_finger],   # up
            [0.0, self.grasp_position, lift_height, 0.0, grasp_finger, grasp_finger],   # rest
        ])
        num_steps = [30, 10, 20, 10, 50, 20]
        actions = []
        for stage in range(target_qs.shape[0] - 1):  # linear interpolation
            for i in range(num_steps[stage]):
                u = target_qs[stage] + (i + 1) * (target_qs[stage + 1] - target_qs[stage]) / num_steps[stage]
                actions.append(u)
        actions = np.array(actions)
        tactile_masks = np.ones(actions.shape[0], dtype=bool)
        tactile_masks[:(num_steps[0] + num_steps[1])] = False
        # self.tactile_samples = tactile_masks.sum()

        #################################################################################
        # qs is the states information of the simulation trajectory
        # qs is (T, ndof_q), each time step, q is a ndof_q-dim vector consisting:
        # qs[t, 0:3]: position of gripper base
        # qs[t, 3]: z-axis rotation of gripper revolute joint
        # qs[t, 4:6]: the positions of two gripper fingers (along x axis)
        # qs[t, 6:9]: the position of the object
        # qs[t, 9:12]: the orientation of the object in rotvec representation
        #################################################################################
        qs, tactiles = self.sim_epi_forward(np.hstack((actions[0], np.zeros(6))), actions, tactile_masks)
        self.obs_buf = tactiles[-1].reshape(self.tactile_samples, 2, self.tactile_rows, self.tactile_cols, 3)[..., 0:2].copy()
        self.obs_buf = self.normalize_tactile(self.obs_buf)
        self.obs_buf = self.obs_buf.reshape(-1)

        # compute reward
        object_rotvec = qs[-1, 9:12]
        nabs_angle = Rotation.from_rotvec(object_rotvec).as_euler('xyz', False)[0]
        self.slide_com(nabs_angle)
        abs_angle = np.abs(nabs_angle)
        if abs_angle < 0.001 and qs[-1, -4] > 0.005:
            success = True
        else:
            success = False

        if success:
            if self.verbose:
                print('Success: ', abs_angle)
            self.reward_buf = 100.
            self.done_buf = True
            self.is_success = True
        else:
            if self.verbose:
                print('Failure: ', abs_angle)
            self.reward_buf = -abs_angle * 10.
            self.done_buf = False
            self.is_success = False

        self.current_q = qs[-1].copy()

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

    # return tactile obs array: shape (T, 2, nrows, ncols, 2)
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
