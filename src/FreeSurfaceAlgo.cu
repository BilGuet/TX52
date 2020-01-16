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

        if (Wmin + 0.2 * (Wmax - Wmin) < eigenValues[i] && eigenValues[i] <= Wmin + 0.75 * (Wmax - Wmin))
        {
            // initialization to one, if find out that not a boundary particle
            // set to 0
            flags[i] = 1;

            for (size_t j = 0; j < n; j++) {
                Point q = points[j];
                double r = sqrt(pow(p.x - q.x, 2) + pow(p.y - q.y, 2));

                if (r <= 2 * p.h && i != j) {
                    double normal_norm = sqrt(pow(normals[i].x, 2) + pow(normals[i].y, 2));
                    Point t { p.x + (normals[i].x/normal_norm)*p.h, p.y + (normals[i].y/normal_norm)*p.h };
                    
                    Vector tau = { normals[i].y, -normals[i].x };
                    double tau_norm = sqrt(pow(tau.x, 2) + pow(tau.y, 2));
                    tau.x /= tau_norm;
                    tau.y /= tau_norm;

                    Vector xji = {q.x - p.x, q.y - p.y};
                    double xji_norm = sqrt(pow(xji.x, 2) + pow(xji.y, 2));

                    Vector xjT = { q.x - t.x, q.y - t.y };
                    double xjT_norm = sqrt(pow(xjT.x, 2) + pow(xjT.y, 2));
                    //if (abs(p.x - q.x) >= sqrt(2 * p.h) &&
                    //    abs(p.y - q.y) >= sqrt(2 * p.h) &&
                    //    abs(p.x - t.x) < p.h &&
                    //    abs(p.y - t.y) <  p.h) {
                    if(xji_norm >= sqrt(2 * p.h) && xjT_norm < p.h) {

                        flags[i] = 0;
                    }

                    //if (abs(p.x - q.x) < sqrt(2 * p.h) &&
                    //    abs(p.y - q.y) < sqrt(2 * p.h) &&
                    //    abs(normals[2*i] * p.x - t.x) + abs(tau[0] * p.x - t.x) < p.h &&
                    //    abs(normals[2*i + 1] * p.y - t.y) + abs(tau[1] * p.y - t.y) < p.h) {
                    double nx = abs(normals[i].x * xjT.x + normals[i].y * xjT.y);
                    double tauxjT = abs(tau.x * xjT.x + tau.y * xjT.y);
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


double distance(Point p1, Point p2){
	return sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p1.y));
}

double norme(double* v){
	return sqrt(v[0]*v[0] + v[1]*v[1]);
}

double dot_product(std::vector<double> vector1, std::vector<double> vector2){
    double sum = 0;
    for (int i = 0; i < 2; i++){
    	sum += (vector1[i])*(vector2[i]);
    }
    return sum;
}

std::vector<double> generate_orthogonal(const std::vector<double>& a) {
    // get some random data
    std::vector<double> b = std::vector<double>(1,2); //generate_random(a.size());

    // find the last non zero entry in a
    // We have to turn the reverse iterator into an iterator via std::prev(rit.base())
    auto IsZero = [] (const double f) -> bool { return f == double(0.0);};
    auto end = std::prev(std::find_if_not(a.crbegin(), a.crend(), IsZero).base());

    // determine the dot product up to end
    double dotProduct = dot_product(a, b);

    // set the value of b so that the inner product is zero
    b[std::distance(a.cbegin(), end)] = - dotProduct / (*end);

    return b;
}


void FreeSurfaceAlgo(const std::vector<Point>& points, std::vector<double>& eigenValues, 
    std::vector<Vector>& normals, std::vector<int>& flags){ // Function that computes from scratch

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

/*
    for(int i=0; i<points.size(); i++)
    {
    	if(0.2*Wmax < eigenValues[i] || eigenValues[i] <= 0.75*Wmax)
        {
            /*
    		// Computation of Renormalization matrix
    		double rMatrix[2][2];
            double dWdx;
            double dWdy;
            double sumResult[2];
            double vi[2];

    		for(int j=0; j < neighbors[i].size(); j++)
            {
    			computeGradiant(points[i], points[neighbors[i][j]], dWdx, dWdy);

                // v : density
    			sumResult[0] += dWdx * points[neighbors[i][j]].v;
    			sumResult[1] += dWdy * points[neighbors[i][j]].v;
    		}

    		vi[0] = -1 * (rMatrix[0][0]*sumResult[0] + rMatrix[0][1]*sumResult[1]);
    		vi[1] = -1 * (rMatrix[1][0]*sumResult[1] + rMatrix[1][1]*sumResult[1]);

            std::vector<double> Ntemp;

    		Ntemp.push_back(vi[0] / norme(vi) );
    		Ntemp.push_back(vi[1] / norme(vi) );
    		ni.push_back(Ntemp);

*/
/*
    		//Normal point selection
    		for(int j=0; j < neighbors[i].size(); j++) {

    			Point t(points[i].x + ni[i][0], points[i].y + ni[i][1]);
    			std::vector<double> tau = generate_orthogonal(ni[i]);

    			if(abs(points[i].x - points[neighbors[i][j]].x) >= sqrt(2*points[i].h) &&
                    abs(points[i].y - points[neighbors[i][j]].y) >= sqrt(2* points[i].h) &&
                    abs(points[i].x-t.x) < sqrt(2*points[i].h) &&
                    abs(points[i].y-t.y) < sqrt(2*points[i].h)) {
    				// Not a normal point
    			}
                else {
    				if(abs(points[i].x - points[neighbors[i][j]].x) < sqrt(2*points[i].h) &&
                        abs(points[i].y - points[neighbors[i][j]].y) < sqrt(2*points[i].h) &&
                        abs(ni[i][0]*points[i].x-t.x) + abs(tau[0]*points[i].x-t.x) < points[i].h &&
                        abs(ni[i][1]*points[i].y-t.y) + abs(tau[1]*points[i].y-t.y) < points[i].h) {
						// Not a normal point
    				}
                    else {
    					// Is a normal point
    					normalPoints.push_back(points[i]);
    				}
    			}
    		}

    	}
        else {
            // not a particle in boundaries
    	}
    }
*/
    free(CPUpoints);
    free(CPUnormals);
    free(CPUeigenValues);
    free(CPUflags);
    cudaFree(GPUpoints);
    cudaFree(GPUnormals);
    cudaFree(GPUeigenValues);
    cudaFree(GPUflags);
}
