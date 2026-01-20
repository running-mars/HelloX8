# HelloX8

## Install JSBSim with Python
```bash
conda create --name sim python=3.8
conda activate sim
pip intall JSBSim
```

## Model setting
The storage path of the UAV dynamics model is as follows:
```bash
~/.conda/envs/sim/lib/python3.8/site-packages/jsbsim/aircraft/
```
So, execute the commnads in the terminal:
```bash
mkdir ~/.conda/envs/sim/lib/python3.8/site-packages/jsbsim/aircraft/x8
cp ./config/x8/x8.xml ~/.conda/envs/sim/lib/python3.8/site-packages/jsbsim/aircraft/x8/
cp ./config/x8/electric800W.xml ~/.conda/envs/sim/lib/python3.8/site-packages/jsbsim/engine/
```