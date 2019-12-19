#pragma once

#include <vector>
#include "Point.h"

void ComputeNormalVector(const std::vector<Point>& points, std::vector< std::vector<double> >& normals, const std::vector<double>& V);
