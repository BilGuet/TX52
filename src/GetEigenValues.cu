#define _USE_MATH_DEFINES

#include <iostream>
#include <math.h>
#include <cassert>
#include "GetEigenValues.cuh"
#include "SolveEigenvalues.h"


void __device__ kernel(const double h, const double r, double xij, double yij, double& dWdx, double& dWdy)
{
    double q = r / h;
    double c1 = 7.0 / (64.0 * M_PI * h * h);
    double c2 = 2.0 * c1 / h;

    if (0.0 < q && q < 2.0)
    {
        dWdx = c2 * (xij / r) * pow(2.0 - q, 3) * (2.0 - q - 2.0 * (2.0 * q + 1.0));
        dWdy = c2 * (yij / r) * pow(2.0 - q, 3) * (2.0 - q - 2.0 * (2.0 * q + 1.0));
    }
    else
    {
        dWdx = 0.0;
        dWdy = 0.0;
    }
}

void __global__ ComputeRenormalizedMatrix(const Point* points, double* L, const size_t n)
{
    // thread computes only particules associates with him
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;

    for (size_t i = id; i < n; i += gridDim.x * blockDim.x)
    {
        const Point p = points[i];

        // matrix elements
        L[4 * i] = 0;       // L[0][0]
        L[4 * i + 1] = 0;   // L[0][1]
        L[4 * i + 2] = 0;   // L[1][0]
        L[4 * i + 3] = 0;   // L[1][1]

        for (size_t j = 0; j < n; j++)
        {
            if (j != i)
            {
                const Point q = points[j];
                double r = sqrt(pow(p.x - q.x, 2) + pow(p.y - q.y, 2));

                // check that q is in support of p for Gaussian kernel
                if (r <= 2* points[i].h)
                {
                    double dWdx = 0;
                    double dWdy = 0;
                    double Vj = points[j].v;
                    double xij = p.x - q.x;
                    double yij = p.y - q.y;

                    // compute dWdx and dWdy
                    kernel(points[i].h, r, xij, yij, dWdx, dWdy);

                    L[4 * i] += (q.x - p.x) * dWdx * Vj;        // L[0][0]
                    L[4 * i + 1] += (q.x - p.x) * dWdy * Vj;    // L[0][1]
                    L[4 * i + 2] += (q.y - p.y) * dWdx * Vj;    // L[1][0]
                    L[4 * i + 3] += (q.y - p.y) * dWdy * Vj;    // L[1][1]
                }
            }
        }
    }
}

void GetEigenValues(const std::vector<Point>& points, std::vector<double>& eigenValues,
    std::vector< std::vector<double> >& matrix)
{
    std::cout << "Computing Eigenvalues..." << std::endl;

    Point* CPUpoints = (Point*)malloc(points.size() * sizeof(Point));
    double* CPUvolumes = (double*)malloc(points.size() * sizeof(double));
    double* CPUmatrix = (double*)malloc(points.size() * 4 * sizeof(double));

    //instanciate CPU variables to send to GPU
    for (int i = 0; i < points.size(); i++)
    {
        CPUpoints[i] = points[i];
        CPUvolumes[i] = points[i].v;
    }


    //GPU variables
    Point* GPUpoints;
    double* GPUmatrix;
    assert(cudaMalloc((void**)&GPUpoints, points.size() * sizeof(Point)) == cudaSuccess);
    assert(cudaMalloc((void**)&GPUmatrix, points.size() * 4 * sizeof(double)) == cudaSuccess);

    assert(cudaMemcpy(GPUpoints, CPUpoints, points.size() * sizeof(Point), cudaMemcpyHostToDevice) == cudaSuccess);

    ComputeRenormalizedMatrix<<< (points.size() / 512) + 1, 512 >>>(GPUpoints, GPUmatrix, points.size());
    cudaDeviceSynchronize();

    assert(cudaMemcpy(CPUmatrix, GPUmatrix, points.size() * 4 * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess);

    matrix.clear();
    eigenValues.clear();
    for (size_t i = 0; i < points.size(); i++)
    {
        // add matrix elements computed in GPU
        matrix.push_back(std::vector<double>());
        matrix[i].push_back(CPUmatrix[4*i]);
        matrix[i].push_back(CPUmatrix[4*i + 1]);
        matrix[i].push_back(CPUmatrix[4*i + 2]);
        matrix[i].push_back(CPUmatrix[4*i + 3]);

        // compute eigenvalues of the matrix
        std::pair<double, double> values = SolveEigenvalues(matrix[i][0], matrix[i][1], matrix[i][2], matrix[i][3]);

        double e1 = values.first;
        double e2 = values.second;

        // save the minimum of the eigenvalues
        double e = e1 < e2 ? e1 : e2;
        eigenValues.push_back(e);
    }

    free(CPUpoints);
    free(CPUvolumes);
    free(CPUmatrix);
    cudaFree(GPUpoints);
    cudaFree(GPUmatrix);
}
