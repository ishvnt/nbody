#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "main.h"
#include "points.h"
#include "curand_kernel.h"

cudaError_t check_error(cudaError_t err);

__global__
void init_galaxy(Point* points, params_t* params);

__global__
void update_pos(Point* points, params_t* params); // euler

__global__
void update_vel(Point* points, params_t* params); // euler

__global__
void update_pos_verlet(Point* points, params_t* params); // verlet

__global__
void update_vel_verlet(Point* points, params_t* params); // verlet

#endif