#include "Point.h"

__global__ void ComputeNeighbors(Point* points, size_t* AllNeighbors, double* distance, size_t n, unsigned int k);
