#include "main.h"
#include "kernels.cuh"

int main()
{
    Point* points;
    check_error( cudaMallocManaged(&points, n_points*sizeof(Point)) );

    int devId;
    check_error( cudaGetDevice(&devId) );

    int num_of_SM;
    check_error( cudaDeviceGetAttribute(&num_of_SM, cudaDevAttrMultiProcessorCount, devId) );

    cudaMemPrefetchAsync(points, n_points*sizeof(Point), devId);

    int num_of_threads = 1024;
    int num_of_blocks = 32 * num_of_SM;

    init_points<<<num_of_blocks, num_of_threads>>>(points);

    check_error( cudaGetLastError() );

    check_error( cudaDeviceSynchronize() );

    Display* win = new Display();
    win->loop(points);
    delete win;

    cudaFree(points);
    return 0;
}