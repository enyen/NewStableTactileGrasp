import sys, os
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(base_dir)
from envs.unstable_grasp_env import UnstableGraspEnv
from examples.UnstableGraspExp.model import CnnFeaEx, TfmerFeaEx


if __name__ == "__main__":
    # training
    if len(sys.argv) == 1:
        # env
        dt = datetime.now().strftime('%m-%d_%H-%M')
        env = make_vec_env(UnstableGraspEnv, n_envs=8, vec_env_cls=SubprocVecEnv)
        env = VecNormalize(env)

        # model
        policy_kwargs = dict(normalize_images=False,
                             # features_extractor_class=CnnFeaEx,
                             # features_extractor_kwargs=dict(features_dim=64),
                             # net_arch=dict(pi=[128, 128], qf=[128, 128]),
                             features_extractor_class=TfmerFeaEx,
                             features_extractor_kwargs=dict(features_dim=32),
                             net_arch=dict(pi=[64, 64], qf=[64, 64]),
                             share_features_extractor=False)
        model = SAC("CnnPolicy", env, gradient_steps=-1, device='cpu',
                    policy_kwargs=policy_kwargs, tensorboard_log='./log')
        # print(model.policy)
        model.learn(total_timesteps=20000, progress_bar=True)
        model.save("./storage/ug_{}_model".format(dt))
        env.save("./storage/ug_{}_stat.pkl".format(dt))

    # testing
    elif len(sys.argv) == 2:
        from matplotlib import pyplot as plt
        import numpy as np

        epi_test = 200
        saved_model = sys.argv[1]
        venv = VecNormalize.load(saved_model + '_stat.pkl', SubprocVecEnv([lambda: UnstableGraspEnv()]))
        venv.training = False
        # env = UnstableGraspEnv(render_style='record')
        # env = UnstableGraspEnv(render_style='loop')
        env = UnstableGraspEnv(render_style='None')
        model = SAC.load(saved_model + "_model", env)
        obs, _ = env.reset()
        env.render()
        lengths, rewards, weight_force, truncation = [], [], [], []

        while True:
            action, _states = model.predict(venv.normalize_obs(obs), deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            if terminated or truncated:
                print('epi:{:03d}, steps:{}, sumR:{:.01f}, lastR:{:.01f}'.format(
                    len(lengths), env.i_steps, env.acc_reward, reward))
                # for recording
                # os.system('ffmpeg {} -filter_complex "[0:v][1:v] concat=n=2:v=1:a=0" -y unstable_grasp.gif'.format(
                #     ' '.join(['-i ./storage/{}.gif'.format(i) for i in range(total_steps + 1)])))
                # os.system('rm ./storage/*.gif')
                # break
                # for stats)
                lengths.append(env.i_steps)
                rewards.append(env.acc_reward)
                weight_force.append([env.weight_weight, env.finger_q])
                truncation.append(truncated)
                env.render()
                obs, _ = env.reset()
                if len(lengths) >= epi_test:
                    break

        succeed = np.logical_not(np.asarray(truncation))
        lengths = np.asarray(lengths)[succeed] * 1.0
        rewards = np.asarray(rewards)[succeed] * 1.0
        weight_force = np.asarray(weight_force)[succeed] * 1.0

        # epi length
        print('Success Rate: {:03f}'.format(succeed.sum() * 1.0 / epi_test))
        print('Epi length: [Min, Avg, Max] = [{}, {:03f}, {}]'.format(
            lengths.min(), lengths.mean(), lengths.max()))

        # weight:force
        lengths = lengths - lengths.min()
        clr = np.stack((lengths / lengths.max(), 1. - (lengths / lengths.max()), np.zeros_like(lengths)), axis=1)
        plt.scatter(weight_force[:, 0] * 1000, weight_force[:, 1] * 1000 + 2.5, c=clr)
        plt.title("Grip force against load weight.")
        plt.xlabel("Load Weight (gram)")
        plt.ylabel("Gripping Distance (mm)")
        plt.xlim(17, 92)
        plt.ylim(0, 5.1)
        plt.show()
