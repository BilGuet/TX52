#pragma once

#include "cuda.h"
#include "cuda_runtime.h"

struct Point
{
    __device__ Point(double _x, double _y) : x(_x), y(_y) {}
    
    double x;
    double y;

    // v : particle volume
    double v;

    // h : spacing inter-partciles
    // use for kernel radius (dx ~ 0.00075)
    //double h = 1.0 * dx;   
    double h = 0.22e-3;
};
