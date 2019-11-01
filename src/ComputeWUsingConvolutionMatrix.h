#pragma once

#include <vector>
#include <cmath>

#include "Point.h"
#include "KNearestNeighbors.cuh"
#include "SolveEigenvalues.h"

void ComputeWUsingConvolutionMatrix(const std::vector<Point>& points, std::vector<double>& W)
{
    std::vector< std::vector<size_t> > neighbors;
    GetKNearestNeighborsGPU(points, neighbors);

    for (size_t p = 0; p < points.size(); p++)
    {
        double Xmean = 0, Ymean = 0;

        for (size_t q = 0; q < k; q++)
        {
            Xmean += points[neighbors[p][q]].x;
            Ymean += points[neighbors[p][q]].y;
        }
        Xmean /= k;
        Ymean /= k;

        //convolution matrix elements
        double CovXX = 0, CovXY = 0, CovYX = 0, CovYY = 0;

        for (size_t q = 0; q < k; q++)
        {
            CovXX += (points[neighbors[p][q]].x - Xmean) * (points[neighbors[p][q]].x - Xmean);
            CovXY += (points[neighbors[p][q]].x - Xmean) * (points[neighbors[p][q]].y - Ymean);
            CovYY += (points[neighbors[p][q]].y - Ymean) * (points[neighbors[p][q]].y - Ymean);
        }

        CovXX /= k;
        CovXY /= k;
        CovYY /= k;
        CovYX = CovXY;

        std::vector<double> values = SolveEigenvalues(CovXX, CovXY, CovYX, CovYY);
        double eigenvalue1 = values[0];
        double eigenvalue2 = values[1];

        if (eigenvalue1 < eigenvalue2)
        {
            W[p] =  eigenvalue1/(eigenvalue1 + eigenvalue2);
        }
        else
        {
            W[p] =  eigenvalue2/(eigenvalue1 + eigenvalue2);
        }
    }
}
