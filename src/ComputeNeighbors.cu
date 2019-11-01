#include <iostream>

#include "Point.h"

__device__ double get_maximum(double* distance, int& index)
{
    double max = 0;

    for(int i = 0; i < k; i++)
    {
        if(distance[i] > max)
        {
            max = distance[i];
            index = i;
        }
    }

    return max;
}

__global__ void ComputeNeighbors(Point* points, size_t* AllNeighbors, size_t n)
{
    // this thread calculate for points associates to it by id
    int id = blockIdx.x*blockDim.x + threadIdx.x;

    // gridDim.x*blockDim.x = number of threads
    for(int p = id; p < n; p += gridDim.x*blockDim.x)
    {
        double distance [k];
        for (int i = 0; i < k; i++)
        {
            distance[i] = 100000;
        }

        for (size_t q = 0; q < n; q ++)
        {
            //check that we're not calculating the distance between p and itself
            if (q != p)
            {
                // calcuate the distance between p and q
                double d = sqrt(pow(points[p].x - points[q].x, 2) + pow(points[p].y - points[q].y, 2));

                int index;
                double max = get_maximum(distance, index);

                if(d < max)
                {
                    distance[index] = d;
                    AllNeighbors[p*k + index] = q;
                }
            }
        }
    }
}
