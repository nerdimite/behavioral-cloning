# Behavioral Cloning

This repository contains the code for the CellStart Webinar on ["Intro to Self Driving Cars"](https://www.meetup.com/Disrupt-4-0/events/269057912/).
 
## Setup
1. Download the [Udacity Car Simulator](https://github.com/udacity/self-driving-car-sim).
2. Create a conda environment with the required libraries.
```bash
 # Without GPU
 conda env create -f car-env.yml

 # With GPU
 conda env create -f car-gpu-env.yml
 ```

## Usage

1. Record Training data with the simulator
2. For Training a new model, run the [Behavioral Cloning](BehavioralCloning.ipynb) notebook
3. After training the model, you can run the `drive.py` script as 
 ```bash
 python drive.py model.h5
 ```
 
### References
 
https://devblogs.nvidia.com/deep-learning-self-driving-cars
https://arxiv.org/abs/1604.07316v1
https://github.com/udacity/CarND-Behavioral-Cloning-P3
https://github.com/udacity/self-driving-car-sim
https://github.com/naokishibuya/car-behavioral-cloning
