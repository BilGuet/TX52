#define _USE_MATH_DEFINES

#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <math.h>
#include <cassert>
#include <utility>
#include "FreeSurfaceAlgo.cuh"
#include "ComputeNormalVector.cuh"

// Rajouter dans classe Point attributs h,v;
// Penser au 2h radius distance pour les voisins
// Variable h non définie
//dot product et genrate_orthogonal not tested

void __global__ FindBoundaryParticles(Point* points, Vector* normals, double* eigenValues, int* flags,
    double Wmin, double Wmax, size_t n) {

    size_t id = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = id; i < n; i+= gridDim.x * blockDim.x)
    {
        Point p = points[i];

        // check if particle isn't inside the domain
        if (Wmin + 0.2 * (Wmax - Wmin) < eigenValues[i] && eigenValues[i] <= Wmin + 0.75 * (Wmax - Wmin))
        {
            // initialization to one, if find out that not a boundary particle
            // set to 0
            flags[i] = 1;

            for (size_t j = 0; j < n; j++) {
                Point q = points[j];
                // distance between p and q
                double r = sqrt(pow(p.x - q.x, 2) + pow(p.y - q.y, 2));

                // check that q is in radius of p
                if (r <= 2 * p.h && i != j) {
                    double normal_norm = sqrt(pow(normals[i].x, 2) + pow(normals[i].y, 2));

                    // point T is at a distance of h from p in the direction of its normal
                    Point t { p.x + (normals[i].x/normal_norm)*p.h, p.y + (normals[i].y/normal_norm)*p.h };
                    
                    // tau is orthogonal to the normal of p
                    Vector tau = { normals[i].y, -normals[i].x };
                    double tau_norm = sqrt(pow(tau.x, 2) + pow(tau.y, 2));
                    // normalize tau to make it a unit vector
                    tau.x /= tau_norm;
                    tau.y /= tau_norm;

                    Vector xji = {q.x - p.x, q.y - p.y};
                    double xji_norm = sqrt(pow(xji.x, 2) + pow(xji.y, 2));

                    Vector xjT = { q.x - t.x, q.y - t.y };
                    double xjT_norm = sqrt(pow(xjT.x, 2) + pow(xjT.y, 2));

                    // first condition
                    if(xji_norm >= sqrt(2 * p.h) && xjT_norm < p.h) {

                        flags[i] = 0;
                    }

                    double nx = abs(normals[i].x * xjT.x + normals[i].y * xjT.y);
                    double tauxjT = abs(tau.x * xjT.x + tau.y * xjT.y);

                    // second condition
                    if (xji_norm < sqrt(2 * p.h) && nx + tauxjT < p.h) {

                        flags[i] = 0;
                    }
                }
            }
        }
        else {
            flags[i] = 0;
        }
    }
}

void FreeSurfaceAlgo(const std::vector<Point>& points, std::vector<double>& eigenValues, 
    std::vector<Vector>& normals, std::vector<int>& flags){ // Function that computes from scratch

    // extract normal of all the points
    std::pair<double,double> dW = ComputeNormalVector(points, eigenValues, normals);
    double Wmin = dW.first;
    double Wmax = dW.second;

    Point* CPUpoints = (Point*)malloc(points.size() * sizeof(Point));
    Vector* CPUnormals = (Vector*)malloc(points.size() * 2 * sizeof(double));
    double* CPUeigenValues= (double*)malloc(points.size() * sizeof(double));
    int* CPUflags = (int*)malloc(points.size() * sizeof(int));

    for (int i = 0; i < points.size(); i++)
    {
        CPUpoints[i] = points[i];
        CPUnormals[i].x = normals[i].x;
        CPUnormals[i].y = normals[i].y;
        CPUeigenValues[i] = eigenValues[i];
    }

    Point* GPUpoints;
    Vector* GPUnormals;
    double* GPUeigenValues;
    int* GPUflags;

    assert(cudaMalloc((void**)&GPUpoints, points.size() * sizeof(Point)) == cudaSuccess);
    assert(cudaMalloc((void**)&GPUnormals, points.size() * sizeof(Vector)) == cudaSuccess);
    assert(cudaMalloc((void**)&GPUeigenValues, points.size() * sizeof(double)) == cudaSuccess);
    assert(cudaMalloc((void**)&GPUflags, points.size() * sizeof(int)) == cudaSuccess);

    assert(cudaMemcpy(GPUpoints, CPUpoints, points.size() * sizeof(Point), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(GPUnormals, CPUnormals, points.size() * sizeof(Vector), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(GPUeigenValues, CPUeigenValues, points.size() * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess);

    std::cout << "Computing algorithm..." << std::endl;
    FindBoundaryParticles<< < (points.size() / 512) + 1, 512 >> >(GPUpoints, GPUnormals, GPUeigenValues, GPUflags, Wmin, Wmax, points.size());
    cudaDeviceSynchronize();

    assert(cudaMemcpy(CPUflags, GPUflags, points.size() * sizeof(int), cudaMemcpyDeviceToHost) == cudaSuccess);

    flags.clear();
    for (size_t i = 0; i < points.size(); i++) {
        flags.push_back(CPUflags[i]);
    }

    free(CPUpoints);
    free(CPUnormals);
    free(CPUeigenValues);
    free(CPUflags);
    cudaFree(GPUpoints);
    cudaFree(GPUnormals);
    cudaFree(GPUeigenValues);
    cudaFree(GPUflags);
}
