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
    if(idx < n_points)
    {
        curandState state;
        curand_init(clock()+idx, 0, 0, &state);
        points[idx].x = curand_uniform_double(&state) * static_cast<double>(SCREEN_WIDTH);
        points[idx].y = curand_uniform_double(&state) * static_cast<double>(SCREEN_HEIGHT);
        printf("point %d : %0.00f, %0.00f\n", idx, points[idx].x, points[idx].y);
    }
}

__global__
void update_vel(Point* points)
{
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    if(idx < n_points)
    {
        double Fx = 0.00f, Fy = 0.00f;
        for(int i=0; i<n_points; i++)
        {
            if(points[idx].x == points[i].x && points[idx].y == points[i].y) continue;
            double dx = points[idx].x - points[i].x;
            double dy = points[idx].y - points[i].y;
            double r = sqrt( (dx * dx) + (dy * dy) );
            double F = ( G * m * m ) / ((r * r) + softening);
            Fx += ( F * dx ) / r;
            Fy += ( F * dy ) / r;
        }
        points[idx].vx -= Fx * dt;
        points[idx].vy -= Fy * dt;
    }
}

__global__
void update_pos(Point* points)
{
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    if(idx < n_points)
    {
        points[idx].x += points[idx].vx * dt;
        points[idx].y += points[idx].vy * dt;
    }
}