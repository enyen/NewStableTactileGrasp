import os
import sys
project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(project_base_dir)
import envs
import gym

if __name__ == '__main__':
    env = gym.make('UnstableGrasp-v1', verbose=True, render_tactile=True)
    action_space = env.action_space
    env.reset()
    for i in range(50):
        action = action_space.sample()
        # action = [0, 0]
        obs, reward, done, _, _ = env.step(action)
        print(reward)
        env.render(mode='once')
        if done:
            print('reset')
            obs = env.reset()
