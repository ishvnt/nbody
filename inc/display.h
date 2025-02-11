#ifndef DISPLAY_H
#define DISPLAY_H

#include <SDL2/SDL.h>
#include "points.h"
#include "kernels.cuh"

constexpr const char* SCREEN_TITLE = "n Body Simulation";

class Display
{
    private:
        SDL_Window* window;
        SDL_Renderer* renderer;
    public:
        Display(int screen_width, int screen_height);
        void loop(Point* points, int n, float dt, float softening, float centre_x, float centre_y);
        ~Display();
};

#endif