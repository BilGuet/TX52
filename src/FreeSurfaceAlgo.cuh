#pragma once

#include <vector>
#include "Vector.h"
#include "Point.h"

void FreeSurfaceAlgo(const std::vector<Point>& points, std::vector<double>& eigenValues, std::vector<Vector>& normals, std::vector<int>& flags);
