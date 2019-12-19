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

void __global__ ComputeRenormalizedMatrix(const Point* points, const double* V, double* L, const size_t n, const double h)
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
                if (r <= 2*h)
                {
                    double dWdx = 0;
                    double dWdy = 0;
                    double Vj = V[j];
                    double xij = p.x - q.x;
                    double yij = p.y - q.y;

                    // compute dWdx and dWdy
                    kernel(h, r, xij, yij, dWdx, dWdy);

                    L[4 * i] += (q.x - p.x) * dWdx * Vj;        // L[0][0]
                    L[4 * i + 1] += (q.x - p.x) * dWdy * Vj;    // L[0][1]
                    L[4 * i + 2] += (q.y - p.y) * dWdx * Vj;    // L[1][0]
                    L[4 * i + 3] += (q.y - p.y) * dWdy * Vj;    // L[1][1]
                }
            }
        }
    }
}


void GetEigenValues(const std::vector<Point>& points, const std::vector<double>& V, const double h,
    std::vector<double>& eigenValues, std::vector< std::vector<double> >& matrix)
{
    std::cout << "Computing Eigenvalues..." << std::endl;
    eigenValues.resize(points.size());

    Point* CPUpoints = (Point*)malloc(points.size() * sizeof(Point));
    double* CPUvolumes = (double*)malloc(points.size() * sizeof(double));
    double* CPUmatrix = (double*)malloc(points.size() * 4 * sizeof(double));

    //instanciate CPU variables to send to GPU
    for (int i = 0; i < points.size(); i++)
    {
        CPUpoints[i] = points[i];
        CPUvolumes[i] = V[i];
    }


    //GPU variables
    Point* GPUpoints;
    double* GPUvolumes;
    double* GPUmatrix;
    assert(cudaMalloc((void**)&GPUpoints, points.size() * sizeof(Point)) == cudaSuccess);
    assert(cudaMalloc((void**)&GPUvolumes, points.size() * sizeof(double)) == cudaSuccess);
    assert(cudaMalloc((void**)&GPUmatrix, points.size() * 4 * sizeof(double)) == cudaSuccess);

    assert(cudaMemcpy(GPUpoints, CPUpoints, points.size() * sizeof(Point), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(GPUvolumes, CPUvolumes, points.size() * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess);

    ComputeRenormalizedMatrix<<< (points.size() / 512) + 1, 512 >>>(GPUpoints, GPUvolumes, GPUmatrix, points.size(), h);
    cudaDeviceSynchronize();

    assert(cudaMemcpy(CPUmatrix, GPUmatrix, points.size() * 4 * sizeof(double), cudaMemcpyDeviceToHost) == cudaSuccess);

    matrix.resize(points.size());
    for (size_t i = 0; i < points.size(); i++)
    {
        matrix[i].resize(4);
        matrix[i][0] = CPUmatrix[4*i];      // M[0][0]
        matrix[i][1] = CPUmatrix[4*i + 1];  // M[0][1]
        matrix[i][2] = CPUmatrix[4*i + 2];  // M[1][0]
        matrix[i][3] = CPUmatrix[4*i + 3];  // M[1][1]

        // compute eigenvalues of the matrix
        std::vector<double> values = SolveEigenvalues(matrix[i][0], matrix[i][1], matrix[i][2], matrix[i][3]);

        // save the minimum of the eigenvalues
        double e1 = values[0];
        double e2 = values[1];

        // save the minimum of the eigenvalues
        eigenValues[i] = e1 < e2 ? e1 : e2;
    }

    free(CPUpoints);
    free(CPUvolumes);
    free(CPUmatrix);
    cudaFree(GPUpoints);
    cudaFree(GPUvolumes);
    cudaFree(GPUmatrix);
}
