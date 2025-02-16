#include <iostream>
#include <getopt.h>
#include "main.h"
#include "display.h"
#include "points.h"
#include "kernels.cuh"

int main(int argc, char *argv[])
{
    int dev_id;
    check_error( cudaGetDevice(&dev_id) );

    int num_of_SM;
    check_error( cudaDeviceGetAttribute(&num_of_SM, cudaDevAttrMultiProcessorCount, dev_id) );

    params_t* params;
    check_error( cudaMallocManaged(&params, sizeof(params_t)) );
    // default parameters
    params->n = 1 << 15;
    params->dt = 0.005f;
    params->softening = 5.00f;
    params->radius = 300.00f;
    params->centre.x = 800.00f;
    params->centre.y = 400.00f;
    params->screen_width = 1600;
    params->screen_height = 800;
    params->dev_id = dev_id;
    params->thread_dim = {1024, 1, 1};
    params->block_dim = {(unsigned int)32 * num_of_SM, 1, 1};

    // change parameters according to command line arguments provided
    if (handle_args(argc, argv, params) == -1)
        return 1;

    Point* points;
    check_error( cudaMallocManaged(&points, (params->n) * sizeof(Point)) );

    Display* win = new Display(params);

    // n/2 points in first galaxy
    init_galaxy<<<params->block_dim, params->thread_dim>>>(points, params->n/2, params->radius, params->centre);
    // n/2 points in second smaller galaxy
    init_galaxy<<<params->block_dim, params->thread_dim>>>(points + (params->n/2), params->n/2, 150.00f, {300.00f, 200.00f});
    check_error( cudaGetLastError() );

    check_error( cudaMemPrefetchAsync(params, sizeof(params_t), cudaCpuDeviceId) );
    check_error( cudaDeviceSynchronize() );

    win->loop(points, params);
    delete win;

    cudaFree(points);
    cudaFree(params);
    return 0;
}

int handle_args(int argc, char* argv[], params_t* params)
{
    int option;
    const char* optstring = "t:n:r:x:y:w:h:";

    while ((option = getopt(argc, argv, optstring)) != -1)
    {
        switch (option)
        {
        case 'n':
            params->n = 1 << atoi(optarg);
            break;
        case 't':
            params->dt = atof(optarg);
            break;
        case 'r':
            params->radius = atof(optarg);
            break;
        case 'x':
            params->centre.x = atof(optarg);
            break;
        case 'y':
            params->centre.y = atof(optarg);
            break;
        case 'w':
            params->screen_width = atoi(optarg);
            params->centre.x = params->screen_width * 0.5f;
            break;
        case 'h':
            params->screen_height = atoi(optarg);
            params->centre.y = params->screen_height * 0.5f;
            break;

        default:
            std::cout << "usage: ./nbody [ARGS]\n\
arguments: \n \
\t-n  number of bodies, calculated as pow(2, n), e.g if n = 10, then number of bodies = pow(2, 10) = 1024 \n \
\t-t  timestep (dt)\n \
\t-x  x-coordinate of centre of galaxy\n \
\t-y  y-coordinate of centre of galaxy\n \
\t-w  width of window\n \
\t-h  height of window\n";
            return -1;
        }
    }
    return 0;
}