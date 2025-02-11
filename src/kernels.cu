#include "main.h"
#include "kernels.cuh"

cudaError_t check_error(cudaError_t err)
{
    if(err != cudaSuccess) printf("error: %s\n", cudaGetErrorString(err));
    return err;
}

__global__
void init_galaxy(Point* points, int n, float radius, float centre_x, float centre_y)
{
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    int stride = blockDim.x * gridDim.x;
    for(int i = idx; i < n; i += stride)
    {
        curandState state;
        curand_init(clock()+idx, 0, 0, &state);

        float theta = curand_uniform(&state) * 2 * M_PI;
        float phi = acosf(2.0f * curand_uniform(&state) - 1.0f); 
        float r = powf(curand_uniform(&state), 1.0f / 3.0f) * radius; 

        points[i].x = centre_x + ( r * sinf(phi) * cosf(theta) );
        
        points[i].y = centre_y + ( r * sinf(phi) * sinf(theta) );
        
        points[i].vx = sinf(theta)*20;
        points[i].vy = -cosf(theta)*20;
        points[i].ax = 0.00f;
        points[i].ay = 0.00f;
    }
}

__global__
void update_vel(Point* points, int n, float dt, float softening, float centre_x, float centre_y)
{
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    int stride = blockDim.x * gridDim.x;
    for(int i = idx; i < n; i += stride)
    {
        float Fx = 0.00f, Fy = 0.00f;
        for(int j = 0; j < n; j++)
        {
            if(i == j)
            {
                float bh_m = 3e4;
                float bh_dx = points[i].x - centre_x;
                float bh_dy = points[i].y - centre_y;
                float bh_rinv = rsqrtf( (bh_dx * bh_dx) + (bh_dy * bh_dy) + softening );
                float bh_F = bh_m * bh_rinv * bh_rinv;
                Fx += bh_F * bh_dx * bh_rinv;
                Fy += bh_F * bh_dy * bh_rinv;
                continue;
            }
            float dx = points[i].x - points[j].x;
            float dy = points[i].y - points[j].y;
            float r_inv = rsqrtf( (dx * dx) + (dy * dy) + softening);
            float F = r_inv*r_inv;
            Fx +=  F * dx  * r_inv;
            Fy +=  F * dy  * r_inv;
        }
        points[i].vx -= Fx * dt * 0.5f;
        points[i].vy -= Fy * dt * 0.5f;
    }
}

__global__
void update_pos(Point* points, int n, float dt)
{
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    int stride = blockDim.x * gridDim.x;
    for(int i = idx; i < n; i += stride)
    {
        points[i].x += points[i].vx * dt;
        points[i].y += points[i].vy * dt;
    }
}

__global__
void update_pos_verlet(Point* points, int n, float dt)
{
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    int stride = blockDim.x * gridDim.x;
    for(int i = idx; i < n; i += stride)
    {
        points[i].x += ( points[i].vx * dt ) + ( 0.5f * points[i].ax * dt * dt );
        points[i].y += ( points[i].vy * dt ) + ( 0.5f * points[i].ay * dt * dt );
    }
}

__global__
void update_vel_verlet(Point* points, int n, float dt, float softening, float centre_x, float centre_y)
{
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    int stride = blockDim.x * gridDim.x;
    for(int i = idx; i < n; i += stride)
    {
        float Fx = 0.00f, Fy = 0.00f;

        points[i].vx += 0.5f * points[i].ax * dt;
        points[i].vy += 0.5f * points[i].ay * dt;

        for(int j = 0; j < n; j++)
        {
            if(i == j)
            {
                float bh_m = 4e4;
                float bh_dx = points[i].x - centre_x;
                float bh_dy = points[i].y - centre_y;
                float bh_rinv = rsqrtf( (bh_dx * bh_dx) + (bh_dy * bh_dy) + softening );
                float bh_F = bh_m * bh_rinv * bh_rinv;
                Fx += bh_F * bh_dx * bh_rinv;
                Fy += bh_F * bh_dy * bh_rinv;
                continue;
            }
            float dx = points[i].x - points[j].x;
            float dy = points[i].y - points[j].y;
            float r_inv = rsqrtf( (dx * dx) + (dy * dy) + softening);
            float F = r_inv*r_inv;
            Fx +=  F * dx  * r_inv;
            Fy +=  F * dy  * r_inv;
        }

        points[i].ax = -Fx;
        points[i].ay = -Fy;

        points[i].vx += 0.5f * points[i].ax * dt;
        points[i].vy += 0.5f * points[i].ay * dt;
    }
}