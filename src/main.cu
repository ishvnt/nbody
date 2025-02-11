#include "main.h"
#include "kernels.cuh"

int main()
{
    Point* points;
    check_error( cudaMallocManaged(&points, n*sizeof(Point)) );

    Display* win = new Display();

    int devId;
    check_error( cudaGetDevice(&devId) );

    int num_of_SM;
    check_error( cudaDeviceGetAttribute(&num_of_SM, cudaDevAttrMultiProcessorCount, devId) );

    cudaMemPrefetchAsync(points, n*sizeof(Point), devId);

    int num_of_threads = 1024;
    int num_of_blocks = 32 * num_of_SM;

    init_points_in_circle<<<num_of_blocks, num_of_threads>>>(points, n, 600, 800, 400);
    //1 init_points_in_circle<<<num_of_blocks, num_of_threads>>>(points+(n/2), n/2, 300, 800, 400);

    check_error( cudaGetLastError() );

    check_error( cudaDeviceSynchronize() );

    win->loop(points);
    delete win;

    cudaFree(points);
    return 0;
}