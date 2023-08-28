from stable_baselines3 import SAC
import sys, os
base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(base_dir)
from datetime import datetime
from envs.unstable_grasp_env import UnstableGraspEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super().__init__(observation_space, features_dim)
        n_c, n_h, n_w = observation_space.shape
        self.model = nn.Sequential(
            nn.Conv2d(n_c, n_c * 2, kernel_size=3),
            nn.BatchNorm2d(n_c * 2),
            nn.SiLU(),
            nn.Conv2d(n_c * 2, n_c * 3, kernel_size=3),
            nn.BatchNorm2d(n_c * 3),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear((n_c * 3) * (n_h - 4) * (n_w - 4), features_dim),
            nn.BatchNorm1d(features_dim),
            nn.SiLU()
        )

    def forward(self, observations):
        return self.model(observations)


if __name__ == "__main__":
    # env
    dt = datetime.now().strftime('%m-%d_%H-%M')
    # env = UnstableGraspEnv()
    env = make_vec_env(UnstableGraspEnv, n_envs=8, vec_env_cls=SubprocVecEnv)
    env = VecNormalize(env)

    # model
    policy_kwargs = dict(features_extractor_class=CustomCNN,
                         features_extractor_kwargs=dict(features_dim=256),
                         net_arch=dict(pi=[128, 64], qf=[128, 64]),
                         normalize_images=False,
                         share_features_extractor=False)
    model = SAC("CnnPolicy", env, gradient_steps=-1, tensorboard_log='./log', policy_kwargs=policy_kwargs)
    # print(model.policy)
    model.learn(total_timesteps=20000, log_interval=20, progress_bar=True)
    model.save("ug_{}".format(dt))

    # # testing
    # env = UnstableGraspEnv()
    # model = SAC.load("ug_08-28_02-39")
    # obs = env.reset()
    # total_steps = 0
    # total_rewards = 0
    # while True:
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     total_steps += 1
    #     total_rewards += reward
    #     if terminated or truncated:
    #         print('total steps: {}, total reward: {}, final reward: {}'.format(total_steps, total_rewards, reward))
    #         env.render()
    #         obs = env.reset()
    #         total_steps = 0
    #         total_rewards = 0
