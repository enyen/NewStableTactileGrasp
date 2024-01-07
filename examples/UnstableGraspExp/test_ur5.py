"""
pip install git+https://github.com/enyen/python-urx
"""
import cv2
import numpy as np
import urx
import time
from marker_flow.marker_flow import MarkerFlow
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
import sys, os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize
base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(base_dir)


class RobUR5:
    def __init__(self, cam_idx=[-1, -1], ip_robot='10.10.10.1'):
        """
        set_tcp with m3d.Orientation.new_euler((a, b, c), 'xyz').rotation_vector
        tcp x: out from tool, y: left of tool, z: forward of tool
        :param ip_robot:
        """
        # robot
        self.rob = urx.Robot(ip_robot, True)
        self.rob.set_tcp((0, 0, 0.23, 1.2092, -1.2092, 1.2092))
        self.rob.set_payload(1.0)
        self.gripper = Robotiq_Two_Finger_Gripper(self.rob)
        self.tactile = MarkerFlow(cam_idx=cam_idx)

        # env param
        self.homej = [1.20595, -1.37872, -2.03732, -1.29741, 1.57073, 1.99222]
        self.grip_height = 0.02
        self.grip_offset = 129
        self.bou_pos = 0.09
        self.bou_width = 4
        self.mul_pos = 0.1
        self.mul_width = 4  # new force control: [25g-100g]->[TODO]; old position control: [13.5mm, 18.5mm],[17g, 92g]->[123, 131]=[grip_offset-mul_width, grip_offset+mul_width]
        self.grip_pos, self.grip_width = 0, 0
        self.num_frame = 11

    def disconnect(self):
        self.rob.close()

    def move_home(self):
        self.rob.movej(self.homej, 0.5, 0.5, wait=True)
        self.move_gripper(0)

    def move_tcp_relative(self, pose, acc=0.5, vel=0.5):
        """
        move eff to relative pose in tool frame with position control
        :param pose: relative differences in [x y z R P Y] (meter, radian)
        :param acc: acceleration
        :param vel: velocity
        """
        if len(pose) == 3:
            pose = pose + [0., 0., 0.]
        self.rob.movel_tool(pose, acc, vel, True)

    def move_gripper(self, val, speed=255, force=255, payload=0.1):
        """
        gripper position control
        :param val: [0-255]
        :param speed: [0-255]
        :param force: [0-255]
        :param payload: kg
        """
        self.gripper.gripper_action(val, speed=speed, force=force, payload=payload)

    def reset(self):
        # reset val
        self.grip_pos = 0
        self.grip_width = 0
        # align load
        self.move_home()
        self.move_gripper(self.grip_offset - 20)
        self.move_tcp_relative([0.05, 0, 0.085])
        self.move_tcp_relative([0, 0, -0.17])
        time.sleep(0.5)
        self.move_tcp_relative([0, 0, 0.17])
        time.sleep(0.5)
        # reset pos
        self.move_home()
        self.move_tcp_relative([0.05, 0, 0])

    def step(self, dx=0, dg=0):
        # action
        grip_pos_ = max(min(self.grip_pos + dx * self.mul_pos, self.bou_pos), -self.bou_pos)
        self.grip_width = max(min(self.grip_width + dg * self.mul_width, self.bou_width), -self.bou_width)

        # grasp
        self.move_tcp_relative([0, 0, grip_pos_ - self.grip_pos])          # pre-grasp
        self.move_gripper(int(self.grip_width + self.grip_offset))         # gripper close
        self.tactile.start()                                               # tactile start
        self.move_tcp_relative([-self.grip_height, 0, 0], 0.1, 0.01)       # move up 2s
        # time.sleep(1.0)                                                    # wait 1.0s
        self.tactile.stop()                                                # tactile stop
        self.move_tcp_relative([self.grip_height, 0, 0])                   # move down
        self.move_gripper(0)                                               # gripper open
        self.grip_pos = grip_pos_
        print(self.grip_pos, self.grip_width)

        # observation
        obs = self.get_obs()
        return obs

    def get_obs(self):
        obs = self.tactile.get_marker_flow()
        obs = obs[5:, 0:1]

        # print(obs.shape)
        # for t in range(obs.shape[0]):
        #     img = self.visualize_tactile(obs[t, 0].transpose([1, 2, 0]))
        #     cv2.imshow("img", img)
        #     cv2.waitKey(100)

        # specific frame in the timestamp
        obs_ = obs[np.linspace(0, obs.shape[0] - 1, self.num_frame).round().astype(int)]

        # averaged frames in the timestamp neighborhood
        # step = obs.shape[0] * 1.0 / self.num_frame
        # obs_ = []
        # for i in range(self.num_frame):
        #     obs_.append(obs[int(np.round(i * step)):int(np.round((i + 1) * step))].mean(axis=0))
        # obs_ = np.stack(obs_, axis=0)
        return obs_

    def visualize_tactile(self, tactile_array, tactile_resolution=50, shear_force_threshold=2):
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


if __name__ == "__main__":
    if len(sys.argv) == 2:
        """
        Run Testing
        python test_ur5.py ./storage/ug_datetime
        """
        # init model, robot
        model = SAC.load(sys.argv[1] + "_model")
        rob = RobUR5(cam_idx=[4, -1])

        # running
        for i in range(3):
            rob.reset()
            obs = rob.step()
            while True:
                action = model.predict(obs, deterministic=True)[0]
                obs = rob.step(*action)
                if input('Terminated (y/n) ?') == 'y':
                    break

        # closing
        rob.move_home()
        rob.disconnect()

    elif len(sys.argv) == 1:
        """
        compute normalization for real tactile sensors
        python test_ur5.py  
        """
        rob = RobUR5(cam_idx=[4, -1])
        obss = []

        # running
        for i in range(3):
            rob.reset()
            obs = rob.step()
            while True:
                action = np.random.uniform(-1, 1, 2)
                obs = rob.step(*action)
                obss.append(obs)
                if input('Terminated (y/n) ?') == 'y':
                    break

        obss = np.concatenate(obss, axis=0)
        obss = obss[:, 0:1]  # left sensor only
        np.save('means', obss.mean(axis=0))
        np.save('stds', obss.std(axis=0))

        # closing
        rob.move_home()
        rob.disconnect()
