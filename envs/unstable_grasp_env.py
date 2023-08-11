import os
import sys
from envs.redmax_torch_functions import EpisodicSimFunction

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
    def __init__(self, use_torch=False, observation_type="tactile_map",
                 render_tactile=False, verbose=False, seed=0):
        asset_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
        model_path = os.path.join(asset_folder, 'unstable_grasp/unstable_grasp.xml')

        self.observation_type = observation_type
        self.use_torch = use_torch
        self.verbose = verbose
        self.render_tactile = render_tactile

        self.tactile_rows = 13
        self.tactile_cols = 10

        if self.observation_type == "tactile_flatten":
            self.ndof_obs = self.tactile_rows * self.tactile_cols * 2 * 2
            self.obs_shape = (self.tactile_rows * self.tactile_cols * 2 * 2,)
        elif self.observation_type == "tactile_map":
            self.obs_shape = (2 * 2, self.tactile_rows, self.tactile_cols)
        else:
            raise NotImplementedError

        self.tactile_samples = 1
        self.tactile_force_buf = torch.zeros(self.obs_shape)
        self.obs_buf = torch.zeros(self.obs_shape)
        self.reward_buf = 0.
        self.done_buf = False
        self.info_buf = {}

        super(UnstableGraspEnv, self).__init__(model_path, seed=seed)

        self.sim.viewer_options.camera_lookat = np.array([0., 0., 1])
        self.sim.viewer_options.camera_pos = np.array([3., -1., 1.7])

        self.qpos_init_reference, self.qvel_init_reference = self.generate_initial_state()

        self.ndof_u = 1
        self.action_space = spaces.Box(low=np.full(self.ndof_u, -1.), high=np.full(self.ndof_u, 1.), dtype=np.float32)
        self.action_scale = 0.05

        self.grasp_position_bound = 0.11

    def _get_obs(self):
        if not self.use_torch:
            return self.obs_buf.detach().cpu().numpy()
        else:
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
        self._progress_step = 0
        self.num_blocks = 27
        self.com_blk = self.np_random.uniform(4, self.num_blocks - 4, 1)
        self.std_blk = self.np_random.uniform(4, 12, 1)
        self.sum_blk = self.np_random.uniform(3000, 7000, 1)
        self.fri_blk = self.np_random.uniform(15, 15, 1)
        self.update_density()

        self.grasp_position = 0.
        self.prev_grasp_position = 0.
        self.current_q = self.qpos_init_reference.clone()

        self.sim.clearBackwardCache()
        self.record_idx += 1
        self.record_episode_idx = 0

        self.grasp()
        obs = self._get_obs()

        # return obs
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
        self._progress_step += 1
        if self.use_torch:
            action = torch.clip(u, -1., 1.)
        else:
            action = torch.clip(torch.tensor(u), -1., 1.)

        action_unnorm = action * self.action_scale

        self.grasp_position = torch.clip(self.grasp_position + action_unnorm[0], -self.grasp_position_bound,
                                         self.grasp_position_bound)

        self.grasp()

        reward, done = self.reward_buf, self.done_buf

        if not self.use_torch:
            reward = reward.detach().cpu().item()

        # if done:
        #     self.render('loop')

        obs = self._get_obs()
        # return obs, reward, done, {'success': self.is_success}
        truncated = False
        return obs, reward, done, truncated, {'success': self.is_success}

    def generate_initial_state(self):
        qpos_init = self.sim.get_q_init().copy()
        grasp_height = 0.2
        qpos_init[2] = grasp_height
        qpos_init[4] = -0.03
        qpos_init[5] = -0.03

        self.sim.set_q_init(qpos_init)

        self.sim.reset(backward_flag=False)

        u = qpos_init[0:6].copy()
        u[2] += 0.003  # hard-coded feedforward term

        self.sim.set_u(u)
        self.sim.forward(500, verbose=False, test_derivatives=False)

        initial_qpos = self.sim.get_q().copy()

        initial_qvel = self.sim.get_qdot().copy()

        return torch.tensor(initial_qpos), torch.tensor(initial_qvel)

    '''
    each grasp consists of five stage:
    (1) move to the target grasp position (achieve by setting the q_init directly)
    (2) close the gripper
    (3) lift and capture the tactile frame
    (4) put down
    (5) open the gripper
    '''

    def grasp(self):
        lift_height = 0.2029862 + 0.03
        grasp_height = 0.2029862
        grasp_finger_position = -0.008

        qpos_init = self.current_q.clone().cpu().numpy()

        qpos_init[1] = self.grasp_position

        self.prev_grasp_position = self.grasp_position

        target_qs = []

        target_qs.append(qpos_init[:6])  # stage 1

        target_qs.append(np.array(
            [0.0, self.grasp_position, grasp_height, 0.0, grasp_finger_position, grasp_finger_position]))  # stage 2
        target_qs.append(np.array(
            [0.0, self.grasp_position, grasp_height, 0.0, grasp_finger_position, grasp_finger_position]))
        target_qs.append(np.array(
            [0.0, self.grasp_position, lift_height, 0.0, grasp_finger_position, grasp_finger_position]))  # stage 3
        target_qs.append(np.array(
            [0.0, self.grasp_position, lift_height, 0.0, grasp_finger_position, grasp_finger_position]))
        target_qs.append(np.array(
            [0.0, self.grasp_position, grasp_height, 0.0, grasp_finger_position, grasp_finger_position]))  # stage 4
        target_qs.append(np.array(
            [0.0, self.grasp_position, grasp_height, 0.0, grasp_finger_position, grasp_finger_position]))
        target_qs.append(np.array(
            [0.0, self.grasp_position, grasp_height, 0.0, qpos_init[4], qpos_init[5]]))  # stage 5

        num_steps = [20, 10, 50, 20, 50, 10, 20]

        assert len(num_steps) == len(target_qs) - 1
        actions = []

        for stage in range(len(target_qs) - 1):
            for i in range(num_steps[stage]):
                u = (target_qs[stage + 1] - target_qs[stage]) / num_steps[stage] * (i + 1) + target_qs[
                    stage]  # linearly interpolate the target joint positions
                actions.append(u)

        actions = torch.tensor(np.array(actions))
        tactile_masks = torch.zeros(actions.shape[0], dtype=bool)
        capture_frame = 60
        tactile_masks[capture_frame] = True

        #################################################################################
        # qs is the states information of the simulation trajectory
        # qs is (T, ndof_q), each time step, q is a ndof_q-dim vector consisting:
        # qs[t, 0:3]: position of gripper base
        # qs[t, 3]: z-axis rotation of gripper revolute joint
        # qs[t, 4:6]: the positions of two gripper fingers (along x axis)
        # qs[t, 6:9]: the position of the object
        # qs[t, 9:12]: the orientation of the object in rotvec representation
        #################################################################################
        # tactiles are the tactile vectors acquired at the time steps specified by tactile_masks
        #################################################################################
        qs, _, tactiles = EpisodicSimFunction.apply(torch.tensor(qpos_init), torch.zeros(self.ndof_q), actions,
                                                    tactile_masks, self.sim, False)

        self.tactile_force_buf = tactiles.reshape(self.tactile_samples, 2, self.tactile_rows, self.tactile_cols, 3)[...,
                                 0:2].clone()

        self.obs_buf = self.tactile_force_buf.clone()

        self.obs_buf = self.normalize_tactile(self.obs_buf)

        if self.observation_type == "tactile_flatten":
            self.obs_buf = self.obs_buf.reshape(-1)
        elif self.observation_type == "tactile_map":
            self.obs_buf = self.obs_buf \
                .permute(0, 1, 4, 2, 3) \
                .reshape(-1, self.tactile_rows, self.tactile_cols)

        # compute reward
        object_rotvec = qs[capture_frame, 9:12]
        abs_angle = Rotation.from_rotvec(object_rotvec).as_euler('xyz', False)[0]
        self.slide_com(abs_angle)
        if np.abs(abs_angle) < 0.01 and qs[capture_frame, -4] > 0.005:
            success = True
        else:
            success = False

        if success:
            if self.verbose:
                print('Success: ', abs_angle)
            self.reward_buf = torch.tensor(100.)
            self.done_buf = True
            self.is_success = True
        else:
            if self.verbose:
                print('Failure: ', abs_angle)
            self.reward_buf = torch.tensor(-np.abs(abs_angle) * 10.)
            self.done_buf = False
            self.is_success = False

        self.current_q = qs[-1].clone()

    '''
    normalize the shear force field
    input: dimension (T, 2, nrows, ncols, 2)
    output: dimension (T, 2, nrows, ncols, 2)
    '''

    def normalize_tactile(self, tactile_arrays):
        normalized_tactile_arrays = tactile_arrays.clone()

        lengths = torch.norm(tactile_arrays, dim=-1)

        max_length = np.max(lengths.numpy()) + 1e-5
        normalized_tactile_arrays = normalized_tactile_arrays / (max_length / 30.)

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
        if self.observation_type == 'tactile_flatten':
            tactile_obs = self.obs_buf.reshape(self.tactile_samples, 2, self.tactile_rows, self.tactile_cols, 2)
        elif self.observation_type == 'tactile_map':
            tactile_obs = self.obs_buf.reshape(self.tactile_samples, 2, 2, self.tactile_rows, self.tactile_cols) \
                .permute(0, 1, 3, 4, 2)

        return tactile_obs.detach().cpu().numpy()

    def get_tactile_forces_array(self):
        return self.tactile_force_buf.detach().cpu().numpy() / 0.000005
