"""
pip install git+https://github.com/enyen/python-urx
"""
import urx
import time
import math3d as m3d
from marker_flow.marker_flow import MarkerFlow
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
import sys
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize
import numpy as np
from einops import rearrange


class RobUR5:
    def __init__(self, ip_robot='192.168.0.5'):
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
        self.tactile = MarkerFlow()

        # env param
        self.homej = [-0.7854, -1.5708, -1.5708, -1.5708, 1.5708, 0]
        self.grip_height = 0.02
        self.grip_width = 100
        # TODO: range
        self.mul_dist = 1
        self.mul_force = 1

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
        #TODO: adjust speed for 2cm/s
        if len(pose) == 3:
            pose = pose + [0, 0, 0]
        self.rob.add_pose_tool(m3d.Transform(pose), acc, vel, True)

    def move_gripper(self, val, speed=255, force=255, payload=0.1):
        """
        gripper position control
        :param val: [0-255]
        :param speed: [0-255]
        :param force: [0-255]
        :param payload: kg
        """
        self.gripper.gripper_action(val, speed=speed, force=force, payload=payload)

    def align_load(self):
        """
        help to align load
        """
        self.move_home()
        self.move_gripper(self.grip_width / 2)
        self.move_tcp_relative([self.grip_height, 0, 0.1])
        for _ in range(4):
            self.move_tcp_relative([0, 0, -0.2])
            self.move_tcp_relative([0, 0, 0.2])
        self.move_home()

    def step(self, dx=0, dg=0):
        self.move_tcp_relative([self.grip_height, 0, dx * self.mul_dist])  # pre-grasp
        # TODO: gripper value convertion
        self.move_gripper(self.grip_width + dg * self.mul_force)           # gripper close
        self.tactile.start()                                               # tactile start
        self.move_tcp_relative([-self.grip_height, 0, 0])                  # move up 1.0s
        time.sleep(0.5)                                                    # wait 0.5s
        self.tactile.stop()                                                # tactile stop
        self.move_tcp_relative([self.grip_height, 0, 0])                   # move down
        self.move_gripper(0)                                               # gripper open
        flow = self.tactile.get_marker_flow()
        flow = self.normalize_flow(flow)
        return flow

    @staticmethod
    def normalize_flow(flow):
        t, s, c, h, w = flow.shape
        flow = rearrange(flow, 't s c h w -> (t s h w) c')
        mag = np.linalg.norm(flow, axis=-1).max()
        flow = flow / ((mag + 1e-5) / 30.)
        flow = rearrange(flow, '(t s h w) c -> t s c h w', t=t, s=s, c=c, h=h, w=w)
        return flow


if __name__ == "__main__":
    """
    python test_ur5.py ./storage/ug_09-16_00-00
    """

    # init model
    model = SAC.load(sys.argv[1] + "_model")
    venv = VecNormalize.load(sys.argv[1] + '_stat.pkl', None)
    venv.training = False

    # init robot
    rob = RobUR5()

    # running
    for i in range(5):
        rob.align_load()
        obs = rob.step()
        while True:
            action = model.predict(venv.normalize_obs(obs), deterministic=True)[0]
            obs = rob.step(*action)
            if input('Terminated (y/n) ?') == 'y':
                break

    # closing
    rob.disconnect()
