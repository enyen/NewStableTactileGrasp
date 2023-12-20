import os
import sys
from os import path
from typing import Callable
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from torch.optim import Adam
base_dir = path.abspath(path.join(path.dirname(path.abspath(__file__)), '../../'))
sys.path.append(base_dir)
from envs.unstable_grasp_env import UnstableGraspEnv
from examples.UnstableGraspExp.model import CnnFeaEx, TfmerFeaEx


if __name__ == "__main__":
    # training
    if len(sys.argv) == 1:
        # env
        dt = datetime.now().strftime('%m-%d_%H-%M')
        env = make_vec_env(UnstableGraspEnv, n_envs=8, vec_env_cls=SubprocVecEnv)

        # model
        def linear_schedule(initial_value):
            def func(progress_remaining):
                return max(3e-4, progress_remaining * initial_value)
            return func
        policy_kwargs = dict(normalize_images=False,
                             optimizer_class=Adam, optimizer_kwargs=dict(betas=(0.9, 0.999), weight_decay=1e-5),
                             features_extractor_class=CnnFeaEx,
                             features_extractor_kwargs=dict(features_dim=64),
                             net_arch=dict(pi=[64, 64], qf=[64, 64]),
                             # features_extractor_class=TfmerFeaEx,
                             # features_extractor_kwargs=dict(features_dim=32),
                             # net_arch=dict(pi=[64, 64], qf=[64, 64]),
                             share_features_extractor=False)
        model = SAC('CnnPolicy', env, device='cpu', learning_starts=1024, gamma=0.995,
                    gradient_steps=-1, target_update_interval=-1, train_freq=(8, 'step'),
                    policy_kwargs=policy_kwargs, tensorboard_log='./log', learning_rate=linear_schedule(1e-3))
        model.learn(total_timesteps=40000, progress_bar=True, tb_log_name=dt)
        model.save('./storage/ug_{}'.format(dt))

    # testing
    elif len(sys.argv) == 3:
        from matplotlib import pyplot as plt
        from sklearn.linear_model import LinearRegression
        import numpy as np

        epi_test = 200
        saved_model = sys.argv[1]
        vis_mode = sys.argv[2]
        assert vis_mode == 'record' or vis_mode == 'show' or vis_mode == 'None'
        env = UnstableGraspEnv(render_style=vis_mode)
        model = SAC.load(saved_model, env)
        obs, _ = env.reset()
        env.render()
        lengths, rewards, weight_force, truncation = [], [], [], []

        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            if terminated or truncated:
                print('epi:{:03d}, steps:{}, sumR:{:.01f}, lastR:{:.01f}'.format(
                    len(lengths), env.i_steps, env.acc_reward, reward))
                if vis_mode == 'record':
                    os.system('ffmpeg {} -filter_complex "[0:v][1:v] concat=n={}:v=1:a=0" -y unstable_grasp.gif'.format(
                        ' '.join(['-i ./storage/{}.gif'.format(i) for i in range(env.i_steps + 1)]), (env.i_steps + 1)))
                    # os.system('rm ./storage/*.gif')
                    break
                # for stats
                lengths.append(env.i_steps)
                rewards.append(env.acc_reward)
                weight_force.append([env.weight_weight, env.finger_q])
                truncation.append(truncated)
                obs, _ = env.reset()
                env.render()
                if len(lengths) >= epi_test:
                    break

        succeed = np.logical_not(np.asarray(truncation))
        lengths = np.asarray(lengths)[succeed] * 1.0
        rewards = np.asarray(rewards)[succeed] * 1.0
        weight_force = np.asarray(weight_force)[succeed] * 1.0

        # epi length
        print('Success Rate: {:03f}'.format(succeed.sum() * 1.0 / epi_test))
        print('Epi length (Min, Avg, Max) = ({}, {:03f}, {})'.format(
            lengths.min(), lengths.mean(), lengths.max()))
        plt.text(0.55, 0.3, 'success: {:.03f}'.format(succeed.sum() * 1.0 / epi_test))

        # weight:force
        rng_weight, rng_force = [25, 100], [0, 5]
        weights = (weight_force[:, 0] * 1000 - rng_weight[0]) / (rng_weight[1] - rng_weight[0])
        forces = (weight_force[:, 1] * 1000 - rng_force[0] + 2.5) / (rng_force[1] - rng_force[0])
        lens = lengths - lengths.min()
        clr = np.stack((lens / lens.max(), 1. - (lens / lens.max()), np.zeros_like(lens)), axis=1)
        plt.scatter(weights, forces, c=clr)
        plt.text(0.36, 0.1, 'Epis length (min, avg, max): {}, {:.03f}, {}'.format(lengths.min(), lengths.mean(), lengths.max()))

        reg = LinearRegression().fit(weights[:, None], forces)
        print('Line gradient: {:.04f}'.format(reg.coef_.squeeze()))
        plt.plot([0, 1], reg.predict([[0], [1]]).flatten())
        plt.text(0.40, 0.2, 'gradient: {:.03f}'.format(reg.coef_.squeeze()))
        plt.text(0.7, 0.2, 'y-intercept: {:.03f}'.format(reg.intercept_.squeeze()))

        plt.title('Grip force against load weight.')
        plt.xlabel('Load Weight')
        plt.ylabel('Gripping Distance')
        plt.xlim(0, 1.05)
        plt.ylim(0, 1.05)
        plt.show()
