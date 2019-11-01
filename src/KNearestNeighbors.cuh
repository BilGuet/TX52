#include <vector>
#include "Point.h"

void cudaCheck ( int status, char* msg );

void GetKNearestNeighborsCPU(const size_t p, const std::vector<Point>& points, std::vector<size_t>& neighbors);

void GetKNearestNeighborsGPU(const std::vector<Point>& points, std::vector< std::vector<size_t> >& AllNeighbors);
