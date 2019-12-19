#include <cassert>
#include <vector>
#include <iostream>
#include "GetParticleSpacing.cuh"

__global__ void ComputeSpacing(const Point* points, double* spacing, size_t n)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = id; i < n; i += gridDim.x * blockDim.x)
    {
        double d = INFINITY;

        const Point p = points[i];
        for (size_t j = 0; j < n; j++)
        {
            if (i != j)
            {
                const Point q = points[j];

                // distance between p and q
                double j_d = sqrt(pow(p.x - q.x, 2) + pow(p.y - q.y, 2));

                // if q is nearer than actual neighbor of p, save the distance between p and q
                d = (j_d < d) ? j_d : d;
            }
        }

        spacing[i] = d;
    }
}

double GetParticleSpacing(const std::vector<Point>& points)
{
    Point* CPUpoints = (Point*)malloc(points.size() * sizeof(Point));

    //instanciate points coordinates
    for (int i = 0; i < points.size(); i++)
    {
        CPUpoints[i] = points[i];
    }

    //GPU variables
    Point* GPUpoints;
    double* GPUspacing;
    assert(cudaMalloc((void**)&GPUpoints, points.size() * sizeof(Point)) == cudaSuccess);
    assert(cudaMalloc((void**)&GPUspacing, points.size() * sizeof(double)) == cudaSuccess);

    //send points coordinates to GPU memory
    assert(cudaMemcpy(GPUpoints, CPUpoints, points.size() * sizeof(Point), cudaMemcpyHostToDevice) == cudaSuccess);

    std::cout << "Computing spacing..." << std::endl;

    ComputeSpacing <<<(points.size() / 512) + 1, 512 >>> (GPUpoints, GPUspacing, points.size());
    // wait until all threads finish computing
    cudaDeviceSynchronize();

    double* CPUspacing = (double*)malloc(points.size() * sizeof(double));

    // send spacing value to CPU
    assert(cudaMemcpy(CPUspacing, GPUspacing, points.size() * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess);

    double dx = 0;

    for (int i = 0; i < points.size(); i++)
    {
        // compute average particules spacing
        dx += CPUspacing[i];
    }
    dx /= points.size();

    free(CPUpoints);
    free(CPUspacing);
    cudaFree(GPUpoints);
    cudaFree(GPUspacing);

    return dx;
}
