#include <stdio.h>
#include "kernels.cuh"

cudaError_t check_error(cudaError_t err)
{
    if (err != cudaSuccess)
        printf("error: %s\n", cudaGetErrorString(err));
    return err;
}

__global__ void init_galaxy(Point *points, params_t *params)
{
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < params->n; i += stride)
    {
        if (i != 0)
        {
            curandState state;
            curand_init(clock() + idx, 0, 0, &state);

            float theta = curand_uniform(&state) * 2 * M_PI;
            float r = expf(-3.0f * curand_uniform(&state)) * params->radius;
            float spiral_angle = theta + 0.5f * r;

            float2 r2 = curand_normal2(&state);

            points[i].pos.x = params->centre.x + (r * cosf(spiral_angle));
            points[i].pos.y = params->centre.y + (r * sinf(spiral_angle));

            float dx = points[i].pos.x - params->centre.x;
            float dy = points[i].pos.y - params->centre.y;
            float d_inv = rsqrtf((dx * dx) + (dy * dy));
            float v = sqrtf(5e5 * d_inv);

            points[i].vel.x = -v * sinf(spiral_angle);
            points[i].vel.y = v * cosf(spiral_angle);
            points[i].acc.x = 0.00f;
            points[i].acc.y = 0.00f;
            points[i].mass = 1;
        }
        else // black hole
        {
            points[i].pos.x = params->centre.x;
            points[i].pos.y = params->centre.y;
            points[i].vel.x = 0.00f;
            points[i].vel.y = 0.00f;
            points[i].acc.x = 0.00f;
            points[i].acc.y = 0.00f;
            points[i].mass = 5e5;
        }
    }
}

__global__ void update_vel(Point *points, params_t *params)
{
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < params->n; i += stride)
    {
        float Fx = 0.00f, Fy = 0.00f;
        for (int j = 0; j < params->n; j++)
        {
            if (i == j)
                continue;

            float dx = points[i].pos.x - points[j].pos.x;
            float dy = points[i].pos.y - points[j].pos.y;
            float r_inv = rsqrtf((dx * dx) + (dy * dy) + params->softening);
            float F = points[i].mass * points[j].mass * r_inv * r_inv;
            Fx += F * dx * r_inv;
            Fy += F * dy * r_inv;
        }
        points[i].vel.x -= (Fx / points[i].mass) * params->dt * 0.5f;
        points[i].vel.y -= (Fy / points[i].mass) * params->dt * 0.5f;
    }
}

__global__ void update_pos(Point *points, params_t *params)
{
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < params->n; i += stride)
    {
        points[i].pos.x += points[i].vel.x * params->dt;
        points[i].pos.y += points[i].vel.y * params->dt;
    }
}

__global__ void update_pos_verlet(Point *points, params_t *params)
{
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < params->n; i += stride)
    {
        points[i].pos.x += (points[i].vel.x * params->dt) + (0.5f * points[i].acc.x * params->dt * params->dt);
        points[i].pos.y += (points[i].vel.y * params->dt) + (0.5f * points[i].acc.y * params->dt * params->dt);
    }
}

__global__ void update_vel_verlet(Point *points, params_t *params)
{
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < params->n; i += stride)
    {
        float Fx = 0.00f, Fy = 0.00f;

        points[i].vel.x += 0.5f * points[i].acc.x * params->dt;
        points[i].vel.y += 0.5f * points[i].acc.y * params->dt;

        for (int j = 0; j < params->n; j++)
        {
            if (i == j)
                continue;

            float dx = points[i].pos.x - points[j].pos.x;
            float dy = points[i].pos.y - points[j].pos.y;
            float r_inv = rsqrtf((dx * dx) + (dy * dy) + params->softening);
            float F = points[i].mass * points[j].mass * r_inv * r_inv;
            Fx += F * dx * r_inv;
            Fy += F * dy * r_inv;
        }

        points[i].acc.x = -Fx / points[i].mass;
        points[i].acc.y = -Fy / points[i].mass;

        points[i].vel.x += 0.5f * points[i].acc.x * params->dt;
        points[i].vel.y += 0.5f * points[i].acc.y * params->dt;
    }
}