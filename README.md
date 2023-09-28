# Tactile-Based Control Environments
Adopted from [source](https://github.com/eanswer/TactileSimulation).

## Installation
1. Clone the project and its submodule from github: `git clone git@github.com:enyen/TactileSimulation.git --recursive `.

2. Install **CMake** >= 3.1.0

3. Create conda environment 

   ```
   conda create -n tactile_sim python=3.9
   conda activate tactile_sim
   pip install torch torchvision opencv-python einops stable_baselines3 tensorboard scipy pyyaml tqdm rich mathplotlib pybind11 math3d=3.4.1 git+https://github.com/enyen/python-urx
   ```

5. Install `DiffRedMax`

   ```
   cd externals/DiffHand/core
   python setup.py install
   ```

## Run the examples

### Tactile Unstable Grasp
Training in simulation.
```commandline
cd examples/UnstableGraspExp
python train_sb3.py
```

Testing in simulation using model saved in _ug_datetime_.
```commandline
cd examples/UnstableGraspExp
python train_sb3.py ./storage/ug_datetime
```

<p align="center">
    <img src="envs/assets/unstable_grasp/unstable_grasp.gif" alt="unstable_grasp" width="500" /></p>

### On real UR5 & Sensor
Build marker flow library (adopted from [source](https://github.com/GelSight/tracking)).
```commandline
cd examples/UnstableGraspExp/marker_flow
make
```
Run on hardware.
```commandline
cd examples/UnstableGraspExp
python test_ur5.py ./storage/ug_datetime
```
