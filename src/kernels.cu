#include <stdio.h>
#include "kernels.cuh"

cudaError_t check_error(cudaError_t err)
{
    if (err != cudaSuccess)
        fprintf(stderr, "error: %s\n", cudaGetErrorString(err));
    return err;
}

__global__
void init_galaxy(Point* points, unsigned int n, float radius, float2 centre)
{
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride)
    {
        // initialise stars
        if (i != 0)
        {
            curandState state;
            curand_init(clock() + i, 0, 0, &state);                  // initialise seed with current time and index to give it some ranomness

            float theta = curand_uniform(&state) * 2 * M_PI;
            float r = expf(-3.0f * curand_uniform(&state)) * radius; // exponential distribution function, range : (0.04979*radius, 1*radius]
            float spiral_angle = theta + 0.5f * r;                   // modify theta according to r to give the stars proper position and velocity distribution

            points[i].pos.x = centre.x + (r * cosf(spiral_angle));
            points[i].pos.y = centre.y + (r * sinf(spiral_angle));

            float dx = points[i].pos.x - centre.x;
            float dy = points[i].pos.y - centre.y;
            float d_inv = rsqrtf((dx * dx) + (dy * dy));
            float v = sqrtf(black_hole_mass * d_inv);                // give stars orbital velocity

            points[i].vel.x = -v * sinf(spiral_angle);
            points[i].vel.y = v * cosf(spiral_angle);

            // add velocity of centre of mass of galaxy (black hole)
            points[i].vel.x += points[0].vel.x;
            points[i].vel.y += points[0].vel.y;
            
            points[i].acc.x = 0.00f;
            points[i].acc.y = 0.00f;

            points[i].mass = 10 * (curand_uniform(&state));
        }
        else // initialise black hole
        {
            points[i].pos.x = centre.x;
            points[i].pos.y = centre.y;
            if(radius < 200.00f) 
            {
                points[i].vel.x = 30.00f;                           // move the smaller galaxy towards the larger one
                points[i].vel.y = 0.00f;
            }
            else
            {
                points[i].vel.x = 0.00f;
                points[i].vel.y = 0.00f;
            }
            points[i].acc.x = 0.00f;
            points[i].acc.y = 0.00f;
            points[i].mass = black_hole_mass;
        }
    }
}

__global__
void update_pos(Point* points, params_t* params)
{
    int tid = threadIdx.x;
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    int stride = blockDim.x * gridDim.x;
    __shared__ Point pts[256];
    for (int i = idx; i < params->n; i += stride)
    {
        // verlet integration
        // r_n = r_n-1 + v_n-1*dt + (1/2)*a_n-1*dt*dt
        pts[tid] = points[i];
        pts[tid].pos.x += (pts[tid].vel.x * params->dt) + (0.5f * pts[tid].acc.x * params->dt * params->dt);
        pts[tid].pos.y += (pts[tid].vel.y * params->dt) + (0.5f * pts[tid].acc.y * params->dt * params->dt);
        points[i] = pts[tid];
    }
}

__global__
void update_vel(Point* points, params_t* params)
{
    int tid = threadIdx.x;
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    int stride = blockDim.x * gridDim.x;
    __shared__ Point pts[256];
    __shared__ params_t d_params;
    d_params = *params;
    __syncthreads();
    for (int i = idx; i < d_params.n; i += stride)
    {
        pts[tid] = points[i];
        float Fx = 0.00f, Fy = 0.00f;

        // verlet integration
        // v_n = v_n-1 + (1/2)*(a_n-1+a_n)*dt
        // add previous acceleration to velocity (a_n-1)
        pts[tid].vel.x += 0.5f * pts[tid].acc.x * d_params.dt;
        pts[tid].vel.y += 0.5f * pts[tid].acc.y * d_params.dt;

        for (int j = 0; j < d_params.n; j++)
        {
            if (i == j) continue;

            float dx = pts[tid].pos.x - points[j].pos.x;
            float dy = pts[tid].pos.y - points[j].pos.y;
            float r_inv = rsqrtf((dx * dx) + (dy * dy) + d_params.softening);
            float F = pts[tid].mass * points[j].mass * r_inv * r_inv;
            Fx += F * dx * r_inv;
            Fy += F * dy * r_inv;
        }

        pts[tid].acc.x = -Fx / pts[tid].mass;
        pts[tid].acc.y = -Fy / pts[tid].mass;

        // add current acceleration to velocity (a_n)
        pts[tid].vel.x += 0.5f * pts[tid].acc.x * d_params.dt;
        pts[tid].vel.y += 0.5f * pts[tid].acc.y * d_params.dt;

        points[i] = pts[tid];
    }
}