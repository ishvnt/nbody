#ifndef MAIN_H
#define MAIN_H

#include <iostream>
#include "display.h"
#include "points.h"
#include "kernels.cuh"

constexpr int n_points = 1<<15;
constexpr float dt = 0.01f;
constexpr float softening = 5e-1;
constexpr float G = 6.6743e-11;

#endif