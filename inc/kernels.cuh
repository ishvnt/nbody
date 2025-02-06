#ifndef KERNELS_H
#define KERNELS_H

#include "main.h"
#include "curand_kernel.h"

cudaError_t check_error(cudaError_t err);

__global__
void init_points(Point* points);

__global__
void update_pos(Point* points);

__global__
void update_vel(Point* points);

#endif