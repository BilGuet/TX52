#include <vector>
#include <algorithm>
#include <iostream>
#include <stdlib.h>

#include "KNearestNeighbors.cuh"
#include "Point.h"

#define Wmax ? ; // Quelle valeur



// Rajouter dans classe Point attributs h,v;
// Penser au 2h radius distance pour les voisins
// Variable h non définie
//dot product et genrate_orthogonal not tested



void kernel(double rij, double xij, double yij, double hij, double &dWdx, double &dWdy){

	//Wendland C2 Quintic
	double q = rij/hij;
	double coef_1 = 7.0/(64.0 * M_PI * hij * hij);
	double coef_2 = 2.0 * coef_1/hij;

	if(q < 2.0){
		if(q < 0.0){
			dWdx = coef_2 * (xij/rij) * pow(2.0 -q,3) * (2.0 - q - 2.0 * (2.0 q + 1.0));
			dWdy = coef_2 * (yij/rij) * pow(2.0 -q,3) * (2.0 - q - 2.0 * (2.0 q + 1.0));
		} else {
			dWdx = 0.0;
			dWdy = 0.0;
		}
	} else {
		dWdx = 0.0;
		dWdy = 0.0;
	}
}

double distance(Point p1, Point p2){
	return sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p1.y));
}

double norme(double* v){
	return sqrt(v[0]*v[0] + v[1]*v[1]);
}

double dot_product(double *vector1, double vector2[]){
    double sum = 0;
    for (int i = 0; i < 2; i++){
    	sum += (vector1[i])*(vector2[i]);
    }
    return sum;
}

std::vector<double> generate_orthogonal(const std::vector<double>& a) {
    // get some random data
    std::vector<double> b = [1,2]; //generate_random(a.size());

    // find the last non zero entry in a
    // We have to turn the reverse iterator into an iterator via std::prev(rit.base())
    auto IsZero = [] (const double f) -> bool { return f == double(0.0);};
    auto end = std::prev(std::find_if_not(a.crbegin(), a.crend(), IsZero).base());

    // determine the dot product up to end
    double dotProduct = dot_product(a, b));

    // set the value of b so that the inner product is zero
    b[std::distance(a.cbegin(), end)] = - dotProduct / (*end);

    return b;
}

void computeGradiant(Point p1, Point p2, double &dWdx, double &dWdy){

	double rij = distance(p1,p2);
	double xij = abs(p1.x - p2.x);
	double yij = abs(p1.y - p2.y);
	double hij = (p1.h + p2.h)/2 ; // hij moyenne des deux ?

	kernel(rij, xij, yij, hij, dWdx, dWdy);
}

void FreeSurfaceAlgo(const std::vector<Point>& points, const std::vector<Point>& normalPoints){ // Function that computes from scratch

	// Comment Importer H, Volume ...

	std::vector<std::std::vector<double>> ni;

	double *dWdx = (double*)malloc(sizeof(double));
	double *dWdy = (double*)malloc(sizeof(double));
	double vi[2];
	double sumResult[2];

	// Get neighbors for each points
	std::vector< std::vector<size_t> > neighbors;
	GetKNearestNeighborsGPU(points, neighbors);

	// Get W for each point 
	auto W = std::vector<double>(points.size(), 0);
    ComputeWUsingConvolutionMatrix(points, W);

    for(int i=0; i<points.size;i++){
    	
    	std::vector<double> Ntemp,

    	if(W[i]<Wmax){

    		// Computation of Renormalization matrix
    		double rMatrix[2][2];

    		for(int j=0;j<neighbors[i];j++){
    			computeGradiant(points[i],neighbors[i][j]);
    			sumResult[0] += dWdx * neighbors[i][j].v;
    			sumResult[1] += dWdy * neighbors[i][j].v;
    		}

    		vi[0] = -1 * (rMatrix[0][0]*sumResult[0] + rMatrix[0][1]*sumResult[1]);
    		vi[1] = -1 * (rMatrix[1][0]*sumResult[1] + rMatrix[1][1]*sumResult[1]);

    		Ntemp.push_back(vi[0] / norme(vi) );
    		Ntemp.push_back(vi[1] / norme(vi) );
    		ni.push_back(Ntemp);


    		//Normal point selection
    		for(int j=0;j<neighbors[i];j++){

    			Point t(points[i].x + ni[i][0], points[i].y + ni[i][1]);
    			std::vector<double> tau = generate_orthogonal(ni[i]);

    			if(abs(points[i].x-neighbors[i][j].x)>=sqrt(2*h) && abs(points[i].y-neighbors[i][j].y)>=sqrt(2*h)
    									  && abs(points[i].x-t.x)<sqrt(2*h) && abs(points[i].y-t.y)<sqrt(2*h)){
    				// Not a normal point
    			} else {
    				if(abs(points[i].x-neighbors[i][j].x)<sqrt(2*h) && abs(points[i].y-neighbors[i][j].y)<sqrt(2*h)
    									 		   && (abs(n[i][0]*points[i].x-t.x)+abs(tau[0]*points[i].x-t.x))<h
    									 		   && (abs(n[i][1]*points[i].y-t.y)+abs(tau[1]*points[i].y-t.y))<h){
						// Not a normal point
    				} else {
    					// Is a normal point
    					normalPoints.push_back(points[i]);
    				}
    			}
    		}

    	} else {
    		//Ntemp.push_back(0);
    		//Ntemp.push_back(0);
    		//ni.push_back(Ntemp);
    	}
    }
}

/*
Void kernel(double rij, double xij, double yij, double hij, double &Wij, double &dWdx, double &dWdy){

	//Wendland C2 Quintic
	double q = rij/hij;
	double coef_1 = 7.0/(64.0 * M_PI * hij * hij);
	double coef_2 = 2.0 * coef_1/hij;

	if(q < 2.0){
		Wij = coef_1 * pow(2.0 - q,4) * (2.0 * q + 1.0);
		if(q < 0.0){
			dWdx = coef_2 * (xij/rij) * pow(2.0 -q,3) * (2.0 - q - 2.0 * (2.0 q + 1.0));
			dWdy = coef_2 * (yij/rij) * pow(2.0 -q,3) * (2.0 - q - 2.0 * (2.0 q + 1.0));
		} else {
			dWdx = 0.0;
			dWdy = 0.0;
		}
	} else {
		Wij = 0.0;
		dWdx = 0.0;
		dWdy = 0.0;
	}
}
*/