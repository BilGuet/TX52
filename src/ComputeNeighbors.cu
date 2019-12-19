#include <iostream>
#include "ComputeNeighbors.cuh"

__device__ double get_maximum(double* distance, int& index, int p, unsigned int k)
{
    double max = 0;

    for(int i = 0; i < k; i++)
    {
        // check if current distance is farer than maximum distance
        if(distance[p*k + i] > max)
        {
            // save the distance and associates index
            max = distance[p*k + i];
            index = i;
        }
    }

    return max;
}

__global__  void ComputeNeighbors(Point* points, size_t* AllNeighbors, double* distance, size_t n, unsigned int k)
{
    // this thread calculate for points associates to
    int id = blockIdx.x*blockDim.x + threadIdx.x;

    // gridDim.x*blockDim.x = number of threads
    for(int p = id; p < n; p += gridDim.x*blockDim.x)
    {
        for (int i = 0; i < k; i++)
        {
            // initialize all distance with inifnity value
            distance[p*k + i] = INFINITY;
        }

        for (size_t q = 0; q < n; q ++)
        {
            // check that we're not calculating the distance between p and itself
            if (q != p)
            {
                // calcuate the distance between p and q
                double d = sqrt(pow(points[p].x - points[q].x, 2) + pow(points[p].y - points[q].y, 2));

                int index;
                double max = get_maximum(distance, index, p, k);

                // if the distance is lower than the biggest distance, insert that point in neighbors
                if(d < max)
                {
                    distance[p*k + index] = d;
                    AllNeighbors[p*k + index] = q;
                }
            }
        }
    }
}
