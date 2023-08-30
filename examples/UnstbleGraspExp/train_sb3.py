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
from einops import rearrange


class CnnFeaEx(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super().__init__(observation_space, features_dim)
        n_t, n_s, n_c, n_h, n_w = observation_space.shape
        self.model = nn.Sequential(
            nn.Conv2d(n_c * n_t, n_c * n_t * 2, kernel_size=3),
            nn.BatchNorm2d(n_c * n_t * 2),
            nn.SiLU(),
            nn.Conv2d(n_c * n_t * 2, n_c * n_t * 3, kernel_size=3),
            nn.BatchNorm2d(n_c * n_t * 3),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear((n_c * n_t * 3) * (n_h - 4) * (n_w - 4), features_dim // n_s),
            nn.SiLU()
        )

    def forward(self, x):
        b, t, s, c, h, w = x.shape
        x = rearrange(x, 'b t s c h w -> (b s) (t c) h w', b=b, t=t, s=s, c=c, h=h, w=w)
        x = self.model(x)
        x = rearrange(x, '(b s) d -> b (s d)', b=b, s=s)
        return x


if __name__ == "__main__":
    # training
    if len(sys.argv) == 1:
        # env
        dt = datetime.now().strftime('%m-%d_%H-%M')
        env = make_vec_env(UnstableGraspEnv, n_envs=8, vec_env_cls=SubprocVecEnv)
        env = VecNormalize(env)

        # model
        policy_kwargs = dict(features_extractor_class=CnnFeaEx,
                             features_extractor_kwargs=dict(features_dim=256),
                             net_arch=dict(pi=[128, 64], qf=[128, 64]),
                             normalize_images=False,
                             share_features_extractor=False)
        model = SAC("CnnPolicy", env, gradient_steps=-1, tensorboard_log='./log', policy_kwargs=policy_kwargs)
        # print(model.policy)
        model.learn(total_timesteps=20000, progress_bar=True)
        model.save("./storage/ug_{}_model".format(dt))
        env.save("./storage/ug_{}_stat.pkl".format(dt))

    # testing
    elif len(sys.argv) == 2:
        saved_model = sys.argv[1]
        env = VecNormalize.load(saved_model + '_stat.pkl', SubprocVecEnv([lambda: UnstableGraspEnv()]))
        model = SAC.load(saved_model + "_model", env)
        obs = env.reset()
        total_steps = 0
        total_rewards = 0
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, info = env.step(action)
            total_steps += 1
            total_rewards += reward
            if terminated:
                print('total steps: {}, total reward: {}, final reward: {}'.format(total_steps, total_rewards, reward))
                env.render()
                obs = env.reset()
                total_steps = 0
                total_rewards = 0
