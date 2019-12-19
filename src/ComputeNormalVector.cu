#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include "ComputeNormalVector.cuh"
#include "GetParticleSpacing.cuh"
#include "GetEigenValues.cuh"

#define SAVECOEFFICIENTS
#include "SaveCoefficientValues.h"


void __global__ ComputeVector(const Point const* points, const double const* V, const double const* eigenValues, const double const* matrix, double* vectors, const size_t n, const double h)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = id; i < n; i += gridDim.x * blockDim.x)
    {
        const Point p = points[i];

        // martix elements
        double L00 = matrix[4 * i];
        double L01 = matrix[4 * i + 1];
        double L10 = matrix[4 * i + 2];
        double L11 = matrix[4 * i + 3];

        // initialize normal vector
        vectors[2*i] = 0;
        vectors[2*i + 1] = 0;

        for (size_t j = 0; j < n; j++)
        {
            const Point q = points[j];

            // distance between the particules
            double r = sqrt(pow(p.x - q.x, 2) + pow(p.y - q.y, 2));

            // check that q is in support of p for Gaussian kernel
            // and we're not on p
            if (r <= 2 * h && i != j)
            {
                double dWdx = 0;
                double dWdy = 0;
                double Vj = V[j];
                double xij = p.x - q.x;
                double yij = p.y - q.y;

                // compute dWdx and dWdy
                kernel(h, r, xij, yij, dWdx, dWdy);

                // see vector formula
                vectors[2*i] += (eigenValues[j] - eigenValues[i]) * (L00*dWdx + L01*dWdy) * Vj;
                vectors[2*i + 1] += (eigenValues[j] - eigenValues[i]) * (L10*dWdx + L11*dWdy) * Vj;
            }
        }
        vectors[2*i] *= -1;
        vectors[2*i + 1] *= -1;

        double norm = sqrt(pow(vectors[2*i], 2) + pow(vectors[2*i + 1], 2));

        // n = v / ||v||
        vectors[2*i] /= norm;
        vectors[2*i + 1] /= norm;
    }
}

void ComputeNormalVector(const std::vector<Point>& points, std::vector< std::vector<double> >& normals, const std::vector<double>& V)
{
    double dx = GetParticleSpacing(points);

    // use for kernel radius (dx ~ 0.00075)
    //double h = 1.0 * dx;   
    double h = 1.0 * 0.00075;   

    std::vector<double> eigenValues;
    std::vector< std::vector<double> > matrix;
    GetEigenValues(points, V, h, eigenValues, matrix);

    SaveCoefficientValues(points, eigenValues, 0);

    /*
    Point* CPUpoints = (Point*)malloc(points.size() * sizeof(Point));
    double* CPUvolumes = (double*)malloc(points.size() * sizeof(double));
    double* CPUeigenValues = (double*)malloc(points.size() * 2 * sizeof(double));
    double* CPUvectors = (double*)malloc(points.size() * 2 * sizeof(double));
    double* CPUmatrix = (double*)malloc(points.size() * 4 * sizeof(double));

    for (int i = 0; i < points.size(); i++)
    {
        CPUpoints[i] = points[i];
        CPUvolumes[i] = V[i];
        CPUeigenValues[i] = eigenValues[i];
        CPUmatrix[4*i] = matrix[i][0];
        CPUmatrix[4*i + 1] = matrix[i][1];
        CPUmatrix[4*i + 2] = matrix[i][2];
        CPUmatrix[4*i + 3] = matrix[i][3];
    }

    //GPU variables
    Point* GPUpoints;
    double* GPUvolumes;
    double* GPUeigenValues;
    double* GPUvectors;
    double* GPUmatrix;
    assert(cudaMalloc((void**)&GPUpoints, points.size() * sizeof(Point)) == cudaSuccess);
    assert(cudaMalloc((void**)&GPUvolumes, points.size() * sizeof(double)) == cudaSuccess);
    assert(cudaMalloc((void**)&GPUeigenValues, points.size() * 2 * sizeof(double)) == cudaSuccess);
    assert(cudaMalloc((void**)&GPUvectors, points.size() * 2 * sizeof(double)) == cudaSuccess);
    assert(cudaMalloc((void**)&GPUmatrix, points.size() * 4 * sizeof(double)) == cudaSuccess);

    assert(cudaMemcpy(GPUpoints, CPUpoints, points.size() * sizeof(Point), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(GPUvolumes, CPUvolumes, points.size() * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(GPUeigenValues, CPUeigenValues, points.size() * 2 * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(GPUmatrix, CPUmatrix, points.size() * 4 * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess);

    ComputeVector << < (points.size() / 512) + 1, 512 >> > (GPUpoints, GPUvolumes, GPUeigenValues, GPUmatrix, GPUvectors, points.size(), h);
    cudaDeviceSynchronize();

    assert(cudaMemcpy(CPUvectors, GPUvectors, points.size() * 2 * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess);

    normals.resize(points.size());
    for (size_t i = 0; i < points.size(); i++)
    {
        normals[i].resize(2);
        normals[i][0] = CPUvectors[2*i];
        normals[i][1] = CPUvectors[2*i + 1];
    }

    free(CPUpoints);
    free(CPUvolumes);
    free(CPUeigenValues);
    free(CPUvectors);
    free(CPUmatrix);
    cudaFree(GPUpoints);
    cudaFree(GPUvolumes);
    cudaFree(GPUeigenValues);
    cudaFree(GPUvectors);
    cudaFree(GPUmatrix);
    */
}

#undef SAVECOEFFICIENTS
