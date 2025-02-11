#include "main.h"
#include "kernels.cuh"

Display::Display(int screen_width, int screen_height)
{
    int err = SDL_Init(SDL_INIT_VIDEO);
    if(err<0)
    {
        std::cerr<<"Error initialising video, error message: "<<SDL_GetError()<<"\n";
        return;
    }
    window = SDL_CreateWindow(  SCREEN_TITLE,
                                SDL_WINDOWPOS_CENTERED,
                                SDL_WINDOWPOS_CENTERED,
                                screen_width,
                                screen_height,
                                SDL_WINDOW_SHOWN  );
    if(window == nullptr)
    {
        std::cerr<<"Error initialising window, error message: "<<SDL_GetError()<<"\n";
        SDL_Quit();
        return;
    }
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if(renderer == nullptr)
    {
        std::cerr<<"Error initialising renderer, error message: "<<SDL_GetError()<<"\n";
        SDL_DestroyWindow(window);
        SDL_Quit();
        return;
    }
}

void Display::loop(Point* points, int n, float dt, float softening, float centre_x, float centre_y)
{
    int devId;
    check_error( cudaGetDevice(&devId) );

    int num_of_SM;
    check_error( cudaDeviceGetAttribute(&num_of_SM, cudaDevAttrMultiProcessorCount, devId) );

    int num_of_threads = 1024;
    int num_of_blocks = 32 * num_of_SM;

    bool exit = false;
    SDL_Event event;
    while(!exit)
    {
        while(SDL_PollEvent(&event) != 0)
        {
            if(event.type == SDL_QUIT) exit = true;
        }

        check_error( cudaMemPrefetchAsync(points, n*sizeof(Point), devId) );

        update_pos_verlet<<<num_of_blocks, num_of_threads>>>(points, n, dt);
        check_error( cudaGetLastError() );

        update_vel_verlet<<<num_of_blocks, num_of_threads>>>(points, n, dt, softening, centre_x, centre_y);
        check_error( cudaGetLastError() );

        check_error( cudaMemPrefetchAsync(points, n*sizeof(Point), cudaCpuDeviceId) );
        check_error( cudaDeviceSynchronize() );

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
        SDL_RenderClear(renderer);
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, SDL_ALPHA_OPAQUE);
        
        for (int i = 0; i < n; i++)
        {
            int x = static_cast<int>(points[i].x);
            int y = static_cast<int>(points[i].y);
            SDL_SetRenderDrawColor(renderer, 255, 255, 0, SDL_ALPHA_OPAQUE);
            SDL_RenderDrawPoint(renderer, x, y);
        }
        SDL_RenderPresent(renderer);
    }
}

Display::~Display()
{
    if(renderer) SDL_DestroyRenderer(renderer);
    if(window) SDL_DestroyWindow(window);
    SDL_Quit();
}