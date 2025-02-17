#include <iostream>
#include "display.h"
#include "kernels.cuh"

Display::Display(params_t* params)
{
    int err = SDL_Init(SDL_INIT_VIDEO);
    if (err < 0)
    {
        std::cerr << "Error initialising video, error message: " << SDL_GetError() << "\n";
        return;
    }
    window = SDL_CreateWindow(SCREEN_TITLE,
                              SDL_WINDOWPOS_CENTERED,
                              SDL_WINDOWPOS_CENTERED,
                              params->screen_width,
                              params->screen_height,
                              SDL_WINDOW_SHOWN);
    if (window == nullptr)
    {
        std::cerr << "Error initialising window, error message: " << SDL_GetError() << "\n";
        SDL_Quit();
        return;
    }
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (renderer == nullptr)
    {
        std::cerr << "Error initialising renderer, error message: " << SDL_GetError() << "\n";
        SDL_DestroyWindow(window);
        SDL_Quit();
        return;
    }
}

// calculate positions and velocities and display in window until it is closed
void Display::loop(Point* points, params_t* params)
{
    dim3 blocks = params->block_dim;
    dim3 threads = params->thread_dim;
    int dev_id = params->dev_id;
    int n = params->n;

    bool exit = false;
    SDL_Event event;
    while (!exit)
    {
        while (SDL_PollEvent(&event) != 0)
        {
            if (event.type == SDL_QUIT)
                exit = true;
        }
        check_error( cudaMemPrefetchAsync(params, sizeof(params_t), dev_id) );
        check_error( cudaMemPrefetchAsync(points, (n) * sizeof(Point), dev_id) );

        update_pos<<<blocks, threads>>>(points, params);
        check_error( cudaGetLastError() );

        update_vel_tiled<<<blocks, threads>>>(points, params);
        check_error( cudaGetLastError() );

        check_error( cudaMemPrefetchAsync(points, (n) * sizeof(Point), cudaCpuDeviceId) );
        check_error( cudaDeviceSynchronize() );

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, SDL_ALPHA_OPAQUE);
        SDL_RenderClear(renderer);
        SDL_SetRenderDrawColor(renderer, 255, 255, 255, SDL_ALPHA_OPAQUE);

        for (int i = 0; i < n; i++)
        {
            int x = static_cast<int>(points[i].pos.x);
            int y = static_cast<int>(points[i].pos.y);

            int b = (points[i].mass / 10) * 255; // change color based on mass, yellow -> low mass, white -> high mass
            SDL_SetRenderDrawColor(renderer, 255, 255, b, SDL_ALPHA_OPAQUE);
            SDL_RenderDrawPoint(renderer, x, y);
        }
        SDL_RenderPresent(renderer);
    }
}

Display::~Display()
{
    if (renderer)
        SDL_DestroyRenderer(renderer);
    if (window)
        SDL_DestroyWindow(window);
    SDL_Quit();
}