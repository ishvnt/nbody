#ifndef MAIN_H
#define MAIN_H

#include <iostream>
#include "display.h"
#include "points.h"
#include "kernels.cuh"

constexpr int n_points = 1<<12;
constexpr double dt = 0.05f;
constexpr double softening = 5e-1;
constexpr double G = 6.6743e-11;

#endif