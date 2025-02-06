#ifndef DISPLAY_H
#define DISPLAY_H

#include <SDL2/SDL.h>
#include "points.h"
#include "kernels.cuh"

constexpr const char* SCREEN_TITLE = "n Body Simulation";
constexpr int SCREEN_WIDTH = 1200;
constexpr int SCREEN_HEIGHT  = 800;

class Display
{
    private:
        SDL_Window* window;
        SDL_Renderer* renderer;
    public:
        Display();
        void loop(Point* points);
        ~Display();
};

#endif