#pragma once

#include "vector"
#include "Point.h"

void GetEigenValues(const std::vector<Point>& points, std::vector<double>& eigenValues, std::vector< std::vector<double> >& matrix);

void __device__ kernel(const double h, const double r, double xij, double yij, double& dWdx, double& dWdy);
