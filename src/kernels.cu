#include "main.h"
#include "kernels.cuh"

cudaError_t check_error(cudaError_t err)
{
    if(err != cudaSuccess) printf("error: %s\n", cudaGetErrorString(err));
    return err;
}

__global__
void init_points(Point* points)
{
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    int stride = blockDim.x * gridDim.x;
    for(int i = idx; i < n_points; i += stride)
    {
        curandState state;
        curand_init(clock()+idx, 0, 0, &state);
        points[i].x = curand_uniform_double(&state) * static_cast<double>(SCREEN_WIDTH);
        points[i].y = curand_uniform_double(&state) * static_cast<double>(SCREEN_HEIGHT);
        printf("point %d : %0.00f, %0.00f\n", idx, points[i].x, points[i].y);
    }
}

__global__
void update_vel(Point* points)
{
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    int stride = blockDim.x * gridDim.x;
    for(int i = idx; i < n_points; i += stride)
    {
        double Fx = 0.00f, Fy = 0.00f;
        for(int j=0; j<n_points; j++)
        {
            if(points[i].x == points[j].x && points[i].y == points[j].y) continue;
            double dx = points[i].x - points[j].x;
            double dy = points[i].y - points[j].y;
            double r = sqrt( (dx * dx) + (dy * dy) );
            double F = ( G * m * m ) / ((r * r) + softening);
            Fx += ( F * dx ) / r;
            Fy += ( F * dy ) / r;
        }
        points[i].vx -= Fx * dt;
        points[i].vy -= Fy * dt;
    }
}

__global__
void update_pos(Point* points)
{
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    int stride = blockDim.x * gridDim.x;
    for(int i = idx; i < n_points; i += stride)
    {
        points[i].x += points[i].vx * dt;
        points[i].y += points[i].vy * dt;
    }
}