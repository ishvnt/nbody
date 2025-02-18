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
            curand_init(clock() + i, 0, 0, &state);                  // initialise seed with current time and index to give it some randomness

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
            points[i].acc.x = 0.00f;
            points[i].acc.y = 0.00f;
            points[i].mass = black_hole_mass;
            
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
        }
    }
}

__global__
void update_pos(Point* points, params_t* params)
{
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < params->n; i += stride)
    {
        // verlet integration
        // r_n = r_n-1 + v_n-1*dt + (1/2)*a_n-1*dt*dt
        Point pi = points[i];
        pi.pos.x += (pi.vel.x * params->dt) + (0.5f * pi.acc.x * params->dt * params->dt);
        pi.pos.y += (pi.vel.y * params->dt) + (0.5f * pi.acc.y * params->dt * params->dt);
        points[i] = pi;
    }
}

__global__
void update_vel(Point* points, params_t* params)
{
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    int stride = blockDim.x * gridDim.x;
    __shared__ params_t d_params;
    d_params = *params;
    __syncthreads();
    for (int i = idx; i < d_params.n; i += stride)
    {
        Point pi = points[i];
        float Fx = 0.00f, Fy = 0.00f;

        // verlet integration
        // v_n = v_n-1 + (1/2)*(a_n-1+a_n)*dt
        // add previous acceleration to velocity (a_n-1)
        pi.vel.x += 0.5f * pi.acc.x * d_params.dt;
        pi.vel.y += 0.5f * pi.acc.y * d_params.dt;

        for (int j = 0; j < d_params.n; j++)
        {
            if (i == j) continue;

            float dx = pi.pos.x - points[j].pos.x;
            float dy = pi.pos.y - points[j].pos.y;
            float r_inv = rsqrtf((dx * dx) + (dy * dy) + d_params.softening);
            float F = pi.mass * points[j].mass * r_inv * r_inv;
            Fx += F * dx * r_inv;
            Fy += F * dy * r_inv;
        }

        pi.acc.x = -Fx / pi.mass;
        pi.acc.y = -Fy / pi.mass;

        // add current acceleration to velocity (a_n)
        pi.vel.x += 0.5f * pi.acc.x * d_params.dt;
        pi.vel.y += 0.5f * pi.acc.y * d_params.dt;

        points[i] = pi;
    }
}

__global__
void update_vel_tiled(Point* points, params_t* params)
{
    int i = threadIdx.x + (blockDim.x * blockIdx.x);
    __shared__ Point pts[TILE_SIZE];
    __shared__ params_t d_params;
    d_params = *params;
    __syncthreads();

    if(i < d_params.n)
    {
        Point pi = points[i];

        pi.vel.x += 0.5f * pi.acc.x * d_params.dt;
        pi.vel.y += 0.5f * pi.acc.y * d_params.dt;

        float Fx = 0.00f, Fy = 0.00f;
        for(int tile = 0; tile<gridDim.x; tile++)
        {
            pts[threadIdx.x] = points[(tile * blockDim.x) + threadIdx.x];
            __syncthreads();
            for(int k = 0; k<blockDim.x; k++)
            {
                Point pj = pts[k];
                int j = (tile * blockDim.x) + k;
                if(j < d_params.n)
                {
                    float dx = pi.pos.x - pj.pos.x;
                    float dy = pi.pos.y - pj.pos.y;
                    float r_inv = rsqrtf((dx * dx) + (dy * dy) + d_params.softening);
                    float F = pi.mass * pj.mass * r_inv * r_inv;
                    Fx += F * dx * r_inv;
                    Fy += F * dy * r_inv;
                }
            }
            __syncthreads();
        }
        pi.acc.x = -Fx / pi.mass;
        pi.acc.y = -Fy / pi.mass;

        pi.vel.x += 0.5f * pi.acc.x * d_params.dt;
        pi.vel.y += 0.5f * pi.acc.y * d_params.dt;

        points[i] = pi;
    }
}