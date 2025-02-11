#ifndef KERNELS_H
#define KERNELS_H

#include "main.h"
#include "curand_kernel.h"

cudaError_t check_error(cudaError_t err);

__global__
void init_random_points(Point* points, int n);

__global__
void init_galaxy(Point* points, int n, float radius, float centre_x, float centre_y);

__global__
void update_pos(Point* points, int n, float dt); // euler

__global__
void update_vel(Point* points, int n, float dt, float softening, float centre_x, float centre_y); // euler

__global__
void update_pos_verlet(Point* points, int n, float dt); // verlet

__global__
void update_vel_verlet(Point* points, int n, float dt, float softening, float centre_x, float centre_y); // verlet

#endif