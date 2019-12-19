#include <vector>
#include "Point.h"

void GetKNearestNeighborsCPU(const size_t p, const std::vector<Point>& points, std::vector<size_t>& neighbors, unsigned int k);

void GetKNearestNeighborsGPU(const std::vector<Point>& points, std::vector< std::vector<size_t> >& AllNeighbors, unsigned int k);
