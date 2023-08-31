# Tactile-Based Control Environments
Adopted from [source](https://github.com/eanswer/TactileSimulation).

## Installation
1. Clone the project and its submodule from github: `git clone git@github.com:enyen/TactileSimulation.git --recursive `.

2. Install **CMake** >= 3.1.0: [official instruction for cmake installation](https://cmake.org/install/)

3. Create conda environment 

   ```
   conda create -n tactile_sim python=3.10
   conda activate tactile_sim
   pip install torch torchvision opencv-python einops stable_baselines3 tensorboard scipy pyyaml
   ```

5. Install `DiffRedMax`

   ```
   cd externals/DiffHand/core
   python setup.py install
   ```

## Run the examples

### Taactile Unstable Grasp

```commandline
cd examples/UnstableGraspExp
# training
python train_sb3.py
# testing using model saved in ug_datetime
python train_sb3.py ./storage/ug_datetime
```

<p align="center">
    <img src="envs/assets/unstable_grasp/unstable_grasp.gif" alt="unstable_grasp" width="500" /></p>
