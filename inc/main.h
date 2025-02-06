#ifndef MAIN_H
#define MAIN_H

#include <iostream>
#include "display.h"
#include "points.h"
#include "kernels.cuh"

constexpr int n_points = 1<<11;
constexpr double dt = 0.05f;
constexpr double softening = 1e-1;
constexpr double G = 6.6743e-11;

#endif