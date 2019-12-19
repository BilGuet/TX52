#include <vector>
#include <algorithm>
#include <iostream>
#include "KNearestNeighbors.cuh"

void ComputeWUsingNumberNeighbor(const std::vector<Point>& points, std::vector<double>& W, unsigned int k)
{
    std::vector< std::vector<size_t> > AllNeighbors;
    
    // compute the neighbors for all the particules
    GetKNearestNeighborsGPU(points, AllNeighbors, k);
    
    std::cout << std::endl << "Computing W..." << std::endl;
    for (size_t p = 0; p < points.size(); ++p)
    {
        for (size_t q : AllNeighbors[p])
        {
            //check if p is in q neighbourhood
            if (std::count(AllNeighbors[q].begin(), AllNeighbors[q].end(), p) > 0)
            {
                W[p] += 1;
            }
        }

        // more neighbors have p in their neighborhood, the closest W is from 1
        W[p] /= k;
    }
}
