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
        points[i].x = curand_uniform(&state) * static_cast<float>(SCREEN_WIDTH);
        points[i].y = curand_uniform(&state) * static_cast<float>(SCREEN_HEIGHT);
    }
}

__global__
void init_points_in_circle(Point* points, float radius, float centre_x, float centre_y)
{
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    int stride = blockDim.x * gridDim.x;
    for(int i = idx; i < n_points; i += stride)
    {
        curandState state;
        curand_init(clock()+idx, 0, 0, &state);

        float rand_x = curand_uniform(&state) * 2;
        points[i].x = centre_x + ( (rand_x-1) * radius );
        
        float max_y = sqrtf((radius*radius) - ((centre_x-points[i].x)*(centre_x-points[i].x)));
        float rand_y = curand_uniform(&state) * 2;
        points[i].y = centre_y + ( (rand_y-1) * max_y );

        points[i].vx = -((points[i].y - centre_y) / centre_y) * 20;              // do this properly
        points[i].vy = ((points[i].x - centre_x) / centre_x) * 20;             // this too
        float dist_from_centre = sqrtf( (points[i].y - centre_y)*(points[i].y - centre_y) + (points[i].x - centre_x)*(points[i].x - centre_x));
        points[i].m = 1e7 + ((1-(dist_from_centre/radius)) * (1e12 - 1e7));     // and this
    }
}

__global__
void update_vel(Point* points)
{
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    int stride = blockDim.x * gridDim.x;
    for(int i = idx; i < n_points; i += stride)
    {
        float Fx = 0.00f, Fy = 0.00f;
        for(int j=0; j<n_points; j++)
        {
            if(points[i].x == points[j].x && points[i].y == points[j].y) continue;
            float dx = points[i].x - points[j].x;
            float dy = points[i].y - points[j].y;
            float r = sqrtf( (dx * dx) + (dy * dy) );
            float F = ( G * points[i].m * points[j].m ) / ((r * r) + softening);
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