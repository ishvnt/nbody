#include "main.h"
#include "kernels.cuh"
#include "getopt.h"

int n = 1<<15;
float dt = 0.01f;
float softening = 5;
float radius = 300.00f;
float centre_x = 800.00f;
float centre_y = 400.00f;
int screen_width = 1600;
int screen_height = 800;

int main(int argc, char* argv[])
{
    int option;
    const char* optstring = "t:n:r:x:y:w:h:";
    while((option = getopt(argc, argv, optstring)) != -1)
    {
        switch (option)
        {
        case 'n':
            n = min(1<< atoi(optarg), n);
            break;
        case 't':
            dt = atof(optarg);
            break;
        case 'r':
            radius = atof(optarg);
            break;
        case 'x':
            centre_x = atof(optarg);
            break;
        case 'y':
            centre_y = atof(optarg);
            break;
        case 'w':
            screen_width = atoi(optarg);
            break;
        case 'h':
            screen_height = atoi(optarg);
            break;
        default:
            std::cout<<"usage: ";
            break;
        }
    }

    Point* points;
    check_error( cudaMallocManaged(&points, n*sizeof(Point)) );

    Display* win = new Display(screen_width, screen_height);

    int devId;
    check_error( cudaGetDevice(&devId) );

    int num_of_SM;
    check_error( cudaDeviceGetAttribute(&num_of_SM, cudaDevAttrMultiProcessorCount, devId) );

    cudaMemPrefetchAsync(points, n*sizeof(Point), devId);

    int num_of_threads = 1024;
    int num_of_blocks = 32 * num_of_SM;

    init_galaxy<<<num_of_blocks, num_of_threads>>>(points, n, radius, centre_x, centre_y);

    check_error( cudaGetLastError() );

    check_error( cudaDeviceSynchronize() );

    win->loop(points, n, dt, softening, centre_x, centre_y);
    delete win;

    cudaFree(points);
    return 0;
}