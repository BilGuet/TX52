#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cassert>
#include <utility>
#include "ComputeNormalVector.cuh"
#include "GetParticleSpacing.cuh"
#include "GetEigenValues.cuh"

void __global__ ComputeVector(const Point const* points, const double const* eigenValues, const double const* matrix, Vector* normals, const size_t n)
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
        normals[i].x = 0;
        normals[i].y = 0;

        for (size_t j = 0; j < n; j++)
        {
            const Point q = points[j];

            // distance between the particules
            double r = sqrt(pow(p.x - q.x, 2) + pow(p.y - q.y, 2));

            // check that q is in support of p for Gaussian kernel
            // and we're not on p
            if (r <= 2 * points[i].h && i != j)
            {
                double dWdx = 0;
                double dWdy = 0;
                double Vj = points[j].v;
                double xij = p.x - q.x;
                double yij = p.y - q.y;

                // compute dWdx and dWdy
                kernel(points[i].h, r, xij, yij, dWdx, dWdy);

                // see vector formula
                normals[i].x += (eigenValues[j] - eigenValues[i]) * (L00*dWdx + L01*dWdy) * Vj;
                normals[i].y += (eigenValues[j] - eigenValues[i]) * (L10*dWdx + L11*dWdy) * Vj;
            }
        }
        normals[i].x *= -1;
        normals[i].y *= -1;

        double norm = sqrt(pow(normals[i].x, 2) + pow(normals[i].y, 2));

        // n = v / ||v||
        normals[i].x /= norm;
        normals[i].y /= norm;
    }
}

std::pair<double, double> ComputeNormalVector(const std::vector<Point>& points, std::vector<double>& eigenValues, std::vector<Vector>& normals)
{
    //double dx = GetParticleSpacing(points);
    //double dx = 0.2e-3;

    double Wmax = 0;
    double Wmin = 1000000000;

    std::vector< std::vector<double> > matrix;
    GetEigenValues(points, eigenValues, matrix);

    Point* CPUpoints = (Point*)malloc(points.size() * sizeof(Point));
    Vector* CPUvectors = (Vector*)malloc(points.size() * sizeof(Vector));
    double* CPUeigenValues = (double*)malloc(points.size() * 2 * sizeof(double));
    double* CPUmatrix = (double*)malloc(points.size() * 4 * sizeof(double));

    for (int i = 0; i < points.size(); i++)
    {
        CPUpoints[i] = points[i];
        CPUeigenValues[i] = eigenValues[i];
        CPUmatrix[4*i] = matrix[i][0];
        CPUmatrix[4*i + 1] = matrix[i][1];
        CPUmatrix[4*i + 2] = matrix[i][2];
        CPUmatrix[4*i + 3] = matrix[i][3];

        Wmax = eigenValues[i] > Wmax ? eigenValues[i] : Wmax;
        Wmin = eigenValues[i] < Wmin ? eigenValues[i] : Wmin;
    }

    //GPU variables
    Point* GPUpoints;
    Vector* GPUvectors;
    double* GPUeigenValues;
    double* GPUmatrix;
    assert(cudaMalloc((void**)&GPUpoints, points.size() * sizeof(Point)) == cudaSuccess);
    assert(cudaMalloc((void**)&GPUvectors, points.size() * sizeof(Vector)) == cudaSuccess);
    assert(cudaMalloc((void**)&GPUeigenValues, points.size() * 2 * sizeof(double)) == cudaSuccess);
    assert(cudaMalloc((void**)&GPUmatrix, points.size() * 4 * sizeof(double)) == cudaSuccess);

    assert(cudaMemcpy(GPUpoints, CPUpoints, points.size() * sizeof(Point), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(GPUeigenValues, CPUeigenValues, points.size() * 2 * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(GPUmatrix, CPUmatrix, points.size() * 4 * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess);

    ComputeVector << < (points.size() / 512) + 1, 512 >> > (GPUpoints, GPUeigenValues, GPUmatrix, GPUvectors, points.size());
    cudaDeviceSynchronize();

    assert(cudaMemcpy(CPUvectors, GPUvectors, points.size() * sizeof(Vector), cudaMemcpyDeviceToHost) == cudaSuccess);

    normals.clear();
    for (size_t i = 0; i < points.size(); i++)
    {
        normals.push_back({ CPUvectors[i].x , CPUvectors[i].y });
    }

    free(CPUpoints);
    free(CPUeigenValues);
    free(CPUvectors);
    free(CPUmatrix);
    cudaFree(GPUpoints);
    cudaFree(GPUeigenValues);
    cudaFree(GPUvectors);
    cudaFree(GPUmatrix);

    return { Wmin, Wmax };
}

