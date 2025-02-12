#include <stdio.h>
#include "kernels.cuh"

cudaError_t check_error(cudaError_t err)
{
    if(err != cudaSuccess) printf("error: %s\n", cudaGetErrorString(err));
    return err;
}

__global__
void init_galaxy(Point* points, params_t* params)
{
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    int stride = blockDim.x * gridDim.x;
    for(int i = idx; i < params->n; i += stride)
    {
        curandState state;
        curand_init(clock()+idx, 0, 0, &state);

        float theta = curand_uniform(&state) * 2 * M_PI;
        float r = expf(-3.0f * curand_uniform(&state)) * params->radius;
        float spiral_angle = theta + 0.5f * r;

        float2 r2 = curand_normal2(&state);

        points[i].x = params->centre_x + (r * cosf(spiral_angle));
        points[i].y = params->centre_y + (r * sinf(spiral_angle));
        
        float dx = points[i].x - params->centre_x;
        float dy = points[i].y - params->centre_y;
        float d_inv = rsqrtf( (dx * dx) + (dy * dy) );
        float v = sqrtf( 5e5 * d_inv );

        float spiral_velocity_factor_x = 0.2f * r;
        float spiral_velocity_factor_y = 0.4f * r;

        points[i].vx = -v * sinf(spiral_angle) + spiral_velocity_factor_x * cosf(spiral_angle);
        points[i].vy = v * cosf(spiral_angle) + spiral_velocity_factor_y * sinf(spiral_angle);
        points[i].ax = 0.00f;
        points[i].ay = 0.00f;
    }
}

__global__
void update_vel(Point* points, params_t* params)
{
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    int stride = blockDim.x * gridDim.x;
    for(int i = idx; i < params->n; i += stride)
    {
        float Fx = 0.00f, Fy = 0.00f;
        for(int j = 0; j < params->n; j++)
        {
            if(i == j)
            {
                float bh_m = 5e5;
                float bh_dx = points[i].x - params->centre_x;
                float bh_dy = points[i].y - params->centre_y;
                float bh_rinv = rsqrtf( (bh_dx * bh_dx) + (bh_dy * bh_dy) + params->softening );
                float bh_F = bh_m * bh_rinv * bh_rinv;
                Fx += bh_F * bh_dx * bh_rinv;
                Fy += bh_F * bh_dy * bh_rinv;
                continue;
            }
            float dx = points[i].x - points[j].x;
            float dy = points[i].y - points[j].y;
            float r_inv = rsqrtf( (dx * dx) + (dy * dy) + params->softening);
            float F = r_inv*r_inv;
            Fx +=  F * dx  * r_inv;
            Fy +=  F * dy  * r_inv;
        }
        points[i].vx -= Fx * params->dt * 0.5f;
        points[i].vy -= Fy * params->dt * 0.5f;
    }
}

__global__
void update_pos(Point* points, params_t* params)
{
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    int stride = blockDim.x * gridDim.x;
    for(int i = idx; i < params->n; i += stride)
    {
        points[i].x += points[i].vx * params->dt;
        points[i].y += points[i].vy * params->dt;
    }
}

__global__
void update_pos_verlet(Point* points, params_t* params)
{
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    int stride = blockDim.x * gridDim.x;
    for(int i = idx; i < params->n; i += stride)
    {
        points[i].x += ( points[i].vx * params->dt ) + ( 0.5f * points[i].ax * params->dt * params->dt );
        points[i].y += ( points[i].vy * params->dt ) + ( 0.5f * points[i].ay * params->dt * params->dt );
    }
}

__global__
void update_vel_verlet(Point* points, params_t* params)
{
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    int stride = blockDim.x * gridDim.x;
    for(int i = idx; i < params->n; i += stride)
    {
        float Fx = 0.00f, Fy = 0.00f;

        points[i].vx += 0.5f * points[i].ax * params->dt;
        points[i].vy += 0.5f * points[i].ay * params->dt;

        for(int j = 0; j < params->n; j++)
        {
            if(i == j)
            {
                float bh_m = 5e5;
                float bh_dx = points[i].x - params->centre_x;
                float bh_dy = points[i].y - params->centre_y;
                float bh_rinv = rsqrtf( (bh_dx * bh_dx) + (bh_dy * bh_dy) + params->softening );
                float bh_F = bh_m * bh_rinv * bh_rinv;
                Fx += bh_F * bh_dx * bh_rinv;
                Fy += bh_F * bh_dy * bh_rinv;
                continue;
            }
            float dx = points[i].x - points[j].x;
            float dy = points[i].y - points[j].y;
            float r_inv = rsqrtf( (dx * dx) + (dy * dy) + params->softening);
            float F = r_inv*r_inv;
            Fx +=  F * dx  * r_inv;
            Fy +=  F * dy  * r_inv;
        }

        points[i].ax = -Fx;
        points[i].ay = -Fy;

        points[i].vx += 0.5f * points[i].ax * params->dt;
        points[i].vy += 0.5f * points[i].ay * params->dt;
    }
}