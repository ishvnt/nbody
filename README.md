# n-Body Simulation
From Wikipedia,
> In physics and astronomy, an n-body simulation is a simulation of a dynamical system of particles, usually under the influence of physical forces, such as gravity.  

This program aims to simulate n number of bodies under the influence of gravity as smoothly as possible. It uses CUDA to parallelize the workload and increase performance, and SDL2 for displaying the simulation.

## Build
- Install build-essential if not already installed  
  ```
  sudo apt install build-essential
  ```
- Install CUDA from [here](https://developer.nvidia.com/cuda-downloads)
- Install SDL2
  ```
  sudo apt install libsdl2-dev
  ```
- Clone this repository
  ```
  git clone https://github.com/ishvnt/nbody.git
  ```
- Build using make
  ```
  cd nbody
  make
  ```
- Have fun!
  ```
  make run
  ```
## ToDo
- [ ] Implement Barnes - Hut algorithm
- [ ] Implement proper weight and initial velocity control
- [ ] Add command line options
