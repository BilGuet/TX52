#include <vector>
#include "Point.h"

void GetKNearestNeighborsGPU(const std::vector<Point>& points, std::vector< std::vector<size_t> >& AllNeighbors, unsigned int k);
