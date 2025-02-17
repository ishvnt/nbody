#ifndef KERNELS_CUH
#define KERNELS_CUH

#include "main.h"
#include "points.h"
#include "curand_kernel.h"

__constant__ float black_hole_mass = 5e5;

cudaError_t check_error(cudaError_t err);

__global__
void init_galaxy(Point* points, unsigned int n, float radius, float2 centre);

__global__
void update_pos(Point* points, params_t* params);

__global__
void update_vel(Point* points, params_t* params);

#endif