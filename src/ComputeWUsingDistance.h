#pragma once

#include <vector>
#include "Point.h"
#include "KNearestNeighbors.cuh"

void ComputeWUsingDistance(const std::vector<Point>& points, std::vector<double>& W)
{
    double Dmax = 0;

    //calculate the distance maximum between 2 points
    for (unsigned int p = 0; p < points.size(); p++)
    {
        std::cout << "\rComputing the distance maximum of the cloud " << (p+1)*100/points.size() << "%";
        for (unsigned int q = p + 1; q < points.size(); q++)
        {
            double d = Point::Distance(points[p], points[q]);
            if (Dmax < d)
            {
                Dmax = d;
            }
        }
    }
    std::cout << std::endl;

    std::vector< std::vector<size_t> > neighbors;
    GetKNearestNeighborsGPU(points, neighbors);

    std::cout << std::endl << "Computing sigma..." << std::endl;
    for (unsigned int p = 0; p < points.size(); p++)
    {
        double dmax = 0;

        //get the maximum distance between p and its neighbors
        for (unsigned int j = 0; j < k; j++)
        {
            double d = Point::Distance(points[p], points[neighbors[p][j]]);
            if (dmax < d)
            {
                dmax = d;
            }
        }

        //W is the local maximum distance by the global maximum distance
        W[p] = dmax/Dmax;
    }
        
}
