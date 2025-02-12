#ifndef DISPLAY_H
#define DISPLAY_H

#include <SDL2/SDL.h>
#include "main.h"
#include "points.h"

constexpr const char* SCREEN_TITLE = "n Body Simulation";

class Display
{
    private:
        SDL_Window* window;
        SDL_Renderer* renderer;
    public:
        Display(params_t* params);
        void loop(Point* points, params_t* params);
        ~Display();
};

#endif