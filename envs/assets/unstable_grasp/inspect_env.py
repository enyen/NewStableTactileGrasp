import redmax_py as redmax
import numpy as np

# sim = redmax.Simulation('./unstable_grasp.xml', verbose=True)
sim = redmax.Simulation('../stable_grasp/stable_grasp.xml', verbose=True)
sim.viewer_options.camera_lookat = np.array([0., 0., 1])
sim.viewer_options.camera_pos = np.array([2., -2., 1.])
sim.viewer_options.loop = False
sim.viewer_options.infinite = False
sim.viewer_options.speed = 0.4
sim.reset(backward_flag=False)
sim.forward(100)
sim.replay()
