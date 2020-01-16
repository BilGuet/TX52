#include <algorithm>
#include <iostream>
#include <cassert>
#include "KNearestNeighbors.cuh"
#include "ComputeNeighbors.cuh"


void GetKNearestNeighborsGPU(const std::vector<Point>& points, std::vector< std::vector<size_t> >& AllNeighbors, unsigned int k)
{
    std::vector<size_t> neighbors;
    
    Point* CPUpoints = (Point*)malloc(points.size() * sizeof(Point));
    size_t* CPUneighbors = (size_t*)malloc(points.size() * k * sizeof(size_t));
    
    // instanciate points coordinates
    for(int i = 0; i < points.size(); i++)
    {
        CPUpoints[i] = points[i];
    }


    // GPU variables
    Point* GPUpoints;
    size_t* GPUneighbors;
    double* GPUdistance;
    assert(cudaMalloc((void**)&GPUpoints, points.size() * sizeof(Point)) == cudaSuccess);
    assert(cudaMalloc((void**)&GPUneighbors, points.size() * k * sizeof(size_t)) == cudaSuccess);
    assert(cudaMalloc((void**)&GPUdistance, points.size()*k * sizeof(double)) == cudaSuccess);


    // send points coordinates to GPU memory
    assert(cudaMemcpy(GPUpoints, CPUpoints, points.size() * sizeof(Point), cudaMemcpyHostToDevice) == cudaSuccess);
    
    std::cout << "Computing neighbors..." << std::endl;
    ComputeNeighbors<<< (points.size()/512)+1, 512 >>>(GPUpoints, GPUneighbors, GPUdistance, points.size(), k);
    cudaDeviceSynchronize();
    
    // recover the neighbors indexes from GPU memory
    assert(cudaMemcpy(CPUneighbors, GPUneighbors, points.size() * k * sizeof(size_t), cudaMemcpyDeviceToHost) == cudaSuccess);

    // make sure that neighbors vector is at good size
    neighbors.resize(k);
    // make sure that AllNeighbors vector is empty
    AllNeighbors.clear();

    for(int i = 0; i < points.size(); i++)
    {
        for(int j = 0; j < k; j++)
        {
            neighbors[j] = CPUneighbors[i*k + j];
        }

        // ad vector of neighbors to vector of all neighbors
        AllNeighbors.push_back(neighbors);
    }


    free(CPUpoints);
    free(CPUneighbors);
    cudaFree(GPUpoints);
    cudaFree(GPUneighbors);
    cudaFree(GPUdistance);
}
