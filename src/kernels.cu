#include "main.h"
#include "kernels.cuh"

cudaError_t check_error(cudaError_t err)
{
    if(err != cudaSuccess) printf("error: %s\n", cudaGetErrorString(err));
    return err;
}

__global__
void init_random_points(Point* points)
{
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    int stride = blockDim.x * gridDim.x;
    for(int i = idx; i < n_points; i += stride)
    {
        curandState state;
        curand_init(clock()+idx, 0, 0, &state);
        points[i].x = curand_uniform_double(&state) * static_cast<double>(SCREEN_WIDTH);
        points[i].y = curand_uniform_double(&state) * static_cast<double>(SCREEN_HEIGHT);
    }
}

__global__
void init_points_in_circle(Point* points, double radius, double centre_x, double centre_y)
{
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    int stride = blockDim.x * gridDim.x;
    for(int i = idx; i < n_points; i += stride)
    {
        curandState state;
        curand_init(clock()+idx, 0, 0, &state);

        double rand_x = curand_uniform_double(&state) * 2;
        points[i].x = centre_x + ( (rand_x-1) * radius );
        
        double max_y = sqrt((radius*radius) - ((centre_x-points[i].x)*(centre_x-points[i].x)));
        double rand_y = curand_uniform_double(&state) * 2;
        points[i].y = centre_y + ( (rand_y-1) * max_y );

        points[i].vx = ((points[i].y - centre_y) / centre_y) * 60;              // do this properly
        points[i].vy = -((points[i].x - centre_x) / centre_x) * 60;             // this too
        points[i].m = 1e7 + (curand_uniform_double(&state) * (1e12 - 1e7));     // and this
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
            double F = ( G * points[i].m * points[j].m ) / ((r * r) + softening);
            Fx += ( F * dx ) / r;
            Fy += ( F * dy ) / r;
        }
        points[i].vx -= (Fx / points[i].m) * dt;
        points[i].vy -= (Fy / points[i].m) * dt;
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