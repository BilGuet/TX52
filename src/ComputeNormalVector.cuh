#pragma once

#include <vector>
#include <utility>
#include "Point.h"
#include "Vector.h"

std::pair<double, double> ComputeNormalVector(const std::vector<Point>& points, std::vector<double>& eigenValues, std::vector<Vector>& normals);
