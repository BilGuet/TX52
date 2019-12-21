#pragma once

#include "cuda.h"
#include "cuda_runtime.h"

struct Point
{
    Point(double _x, double _y) : x(_x), y(_y) {}
    
    double x;
    double y;
};
