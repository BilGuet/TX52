#pragma once

#include "Point.h"
#include <vector>

void ComputeWUsingNumberNeighbor(const std::vector<Point>& points, std::vector<double>& W, unsigned int k);
